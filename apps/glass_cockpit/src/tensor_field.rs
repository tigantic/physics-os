// Phase 2: Tensor Field Infrastructure
// QTT tensor representation and visualization data structures
// Constitutional compliance: Doctrine 7 (QTT format), Doctrine 8 (compressed storage)

// Phase 3-5 scaffolding: Tensor field data structures for QTT visualization
// Will be integrated with tensor renderer and colormap pipelines

use glam::{Vec3, Vec4};

#[allow(dead_code)]
/// Tensor field grid for atmospheric visualization
#[derive(Debug, Clone)]
pub struct TensorField {
    /// Grid dimensions (width, height, depth)
    pub dimensions: (u32, u32, u32),
    
    /// Tensor data (flattened 3D grid of 3x3 tensors)
    /// Stored as QTT-compressed format in Phase 2+
    pub data: Vec<Tensor3x3>,
    
    /// Visualization parameters
    pub vis_params: VisualizationParams,
}

/// 3x3 symmetric tensor representation
#[derive(Debug, Clone, Copy)]
pub struct Tensor3x3 {
    // Store only 6 unique components (symmetric matrix)
    // [0][0], [0][1], [0][2]
    //         [1][1], [1][2]
    //                 [2][2]
    pub components: [f32; 6],
}

// Phase 3-5 scaffolding: Tensor3x3 operations for field visualization
#[allow(dead_code)]
impl Tensor3x3 {
    pub fn zero() -> Self {
        Self { components: [0.0; 6] }
    }
    
    pub fn from_components(xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32) -> Self {
        Self { components: [xx, xy, xz, yy, yz, zz] }
    }
    
    /// Get tensor component [i][j]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        match (i, j) {
            (0, 0) => self.components[0],
            (0, 1) | (1, 0) => self.components[1],
            (0, 2) | (2, 0) => self.components[2],
            (1, 1) => self.components[3],
            (1, 2) | (2, 1) => self.components[4],
            (2, 2) => self.components[5],
            _ => 0.0,
        }
    }
    
    /// Compute trace (sum of diagonal elements)
    pub fn trace(&self) -> f32 {
        self.components[0] + self.components[3] + self.components[5]
    }
    
    /// Compute Frobenius norm
    pub fn frobenius_norm(&self) -> f32 {
        let sum_sq = self.components.iter().map(|&x| x * x).sum::<f32>();
        sum_sq.sqrt()
    }
    
    /// Get dominant eigenvector (approximate via power iteration)
    pub fn dominant_eigenvector(&self) -> Vec3 {
        let mut v = Vec3::new(1.0, 0.0, 0.0);
        
        // 5 power iterations (sufficient for visualization)
        for _ in 0..5 {
            let ax = Vec3::new(
                self.get(0, 0) * v.x + self.get(0, 1) * v.y + self.get(0, 2) * v.z,
                self.get(1, 0) * v.x + self.get(1, 1) * v.y + self.get(1, 2) * v.z,
                self.get(2, 0) * v.x + self.get(2, 1) * v.y + self.get(2, 2) * v.z,
            );
            let norm = ax.length();
            v = if norm > 1e-6 { ax / norm } else { Vec3::X };
        }
        
        v
    }
}

/// Visualization parameters for tensor field display
#[derive(Debug, Clone)]
pub struct VisualizationParams {
    /// Color mapping mode
    pub color_mode: ColorMode,
    
    /// Intensity scale factor
    pub intensity_scale: f32,
    
    /// Minimum tensor magnitude threshold
    pub threshold: f32,
    
    /// Show tensor glyphs (ellipsoids)
    pub show_glyphs: bool,
    
    /// Show eigenvector field
    pub show_vectors: bool,
}

impl Default for VisualizationParams {
    fn default() -> Self {
        Self {
            color_mode: ColorMode::Magnitude,
            intensity_scale: 1.0,
            threshold: 0.01,
            show_glyphs: false,
            show_vectors: true,
        }
    }
}

// Phase 3-5 scaffolding: Tensor visualization color modes
#[allow(dead_code)]
/// Color mapping modes for tensor visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Color by tensor magnitude (Frobenius norm)
    Magnitude,
    /// Color by trace (pressure-like quantity)
    Trace,
    /// Color by dominant eigenvector direction
    Direction,
    /// Custom heat map
    Heatmap,
}

// Phase 3-5 scaffolding: TensorField methods for QTT visualization integration
#[allow(dead_code)]
impl TensorField {
    /// Create empty tensor field with given dimensions
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        let total_cells = (width * height * depth) as usize;
        Self {
            dimensions: (width, height, depth),
            data: vec![Tensor3x3::zero(); total_cells],
            vis_params: VisualizationParams::default(),
        }
    }
    
    /// Get tensor at grid position
    pub fn get_tensor(&self, x: u32, y: u32, z: u32) -> Option<&Tensor3x3> {
        let (w, h, d) = self.dimensions;
        if x < w && y < h && z < d {
            let index = (z * h * w + y * w + x) as usize;
            self.data.get(index)
        } else {
            None
        }
    }
    
    /// Set tensor at grid position
    pub fn set_tensor(&mut self, x: u32, y: u32, z: u32, tensor: Tensor3x3) {
        let (w, h, d) = self.dimensions;
        if x < w && y < h && z < d {
            let index = (z * h * w + y * w + x) as usize;
            if let Some(cell) = self.data.get_mut(index) {
                *cell = tensor;
            }
        }
    }
    
    /// Generate test pattern (Phase 2 stub - will be replaced with QTT decompression)
    pub fn generate_test_pattern(&mut self) {
        let (w, h, d) = self.dimensions;
        
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    // Create rotating field pattern
                    let fx = (x as f32 / w as f32) * 2.0 - 1.0;
                    let fy = (y as f32 / h as f32) * 2.0 - 1.0;
                    let fz = (z as f32 / d as f32) * 2.0 - 1.0;
                    
                    let r = (fx * fx + fy * fy).sqrt();
                    let intensity = (r * std::f32::consts::PI).cos() * 0.5 + 0.5;
                    
                    let tensor = Tensor3x3::from_components(
                        intensity,           // xx
                        fx * fy * 0.3,       // xy
                        fx * fz * 0.2,       // xz
                        intensity * 0.8,     // yy
                        fy * fz * 0.2,       // yz
                        intensity * 0.6,     // zz
                    );
                    
                    self.set_tensor(x, y, z, tensor);
                }
            }
        }
    }
    
    /// Prepare visualization data for GPU upload
    pub fn prepare_gpu_data(&self) -> Vec<Vec4> {
        // Pack tensor data for shader consumption
        // Each tensor → 2x Vec4 (6 components + magnitude + trace)
        let mut gpu_data = Vec::with_capacity(self.data.len() * 2);
        
        for tensor in &self.data {
            let magnitude = tensor.frobenius_norm();
            let trace = tensor.trace();
            
            // First Vec4: (xx, xy, xz, yy)
            gpu_data.push(Vec4::new(
                tensor.components[0],
                tensor.components[1],
                tensor.components[2],
                tensor.components[3],
            ));
            
            // Second Vec4: (yz, zz, magnitude, trace)
            gpu_data.push(Vec4::new(
                tensor.components[4],
                tensor.components[5],
                magnitude,
                trace,
            ));
        }
        
        gpu_data
    }
    
    /// Get field statistics
    pub fn statistics(&self) -> FieldStatistics {
        let mut max_magnitude = 0.0f32;
        let mut min_magnitude = f32::MAX;
        let mut sum_magnitude = 0.0f32;
        let mut max_trace = f32::MIN;
        let mut min_trace = f32::MAX;
        
        for tensor in &self.data {
            let magnitude = tensor.frobenius_norm();
            let trace = tensor.trace();
            
            max_magnitude = max_magnitude.max(magnitude);
            min_magnitude = min_magnitude.min(magnitude);
            sum_magnitude += magnitude;
            max_trace = max_trace.max(trace);
            min_trace = min_trace.min(trace);
        }
        
        FieldStatistics {
            max_magnitude,
            min_magnitude,
            mean_magnitude: sum_magnitude / self.data.len() as f32,
            max_trace,
            min_trace,
        }
    }
}

#[allow(dead_code)]
/// Tensor field statistics for display
#[derive(Debug, Clone, Copy)]
pub struct FieldStatistics {
    pub max_magnitude: f32,
    pub min_magnitude: f32,
    pub mean_magnitude: f32,
    pub max_trace: f32,
    pub min_trace: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor3x3::from_components(1.0, 0.5, 0.3, 2.0, 0.4, 3.0);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 2.0);
        assert_eq!(t.get(2, 2), 3.0);
        assert_eq!(t.get(0, 1), 0.5);
        assert_eq!(t.get(1, 0), 0.5); // Symmetry
    }

    #[test]
    fn test_tensor_trace() {
        let t = Tensor3x3::from_components(1.0, 0.0, 0.0, 2.0, 0.0, 3.0);
        assert_eq!(t.trace(), 6.0);
    }

    #[test]
    fn test_field_creation() {
        let field = TensorField::new(10, 10, 5);
        assert_eq!(field.dimensions, (10, 10, 5));
        assert_eq!(field.data.len(), 500);
    }

    #[test]
    fn test_field_get_set() {
        let mut field = TensorField::new(5, 5, 5);
        let tensor = Tensor3x3::from_components(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        
        field.set_tensor(2, 2, 2, tensor);
        let retrieved = field.get_tensor(2, 2, 2).unwrap();
        
        assert_eq!(retrieved.components, tensor.components);
    }
}

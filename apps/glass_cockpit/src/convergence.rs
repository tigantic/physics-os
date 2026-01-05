//! Convergence Zone Data Module
//!
//! Phase 6: Probabilistic convergence field visualization
//!
//! Reads probability field from RAM bridge and provides:
//! - Scalar intensity values per grid cell
//! - Temporal probability curves per high-intensity node
//! - Zone detection for hover/click interaction
//!
//! Constitutional: All computation on E-cores, zero P-core interruption

use glam::{Vec2, Vec3};

/// Configuration for convergence field visualization
#[derive(Debug, Clone, Copy)]
pub struct ConvergenceConfig {
    /// Grid resolution (width × height)
    pub resolution: (u32, u32),
    /// Minimum intensity threshold for visibility (0.0 - 1.0)
    pub visibility_threshold: f32,
    /// Intensity threshold for "high probability" zones (0.0 - 1.0)
    pub high_intensity_threshold: f32,
    /// Pulse frequency multiplier for high-intensity zones
    pub pulse_frequency: f32,
    /// Maximum cells to render (budget constraint)
    /// Phase 7: Used for LOD-based budget limiting
    #[allow(dead_code)] // Phase 7: LOD budget
    pub max_cells: u32,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            resolution: (128, 64), // Lat/lon grid
            visibility_threshold: 0.05, // Lowered from 0.1 for more visible cells
            high_intensity_threshold: 0.7,
            pulse_frequency: 2.0,
            max_cells: 8192, // Conservative for 60 FPS
        }
    }
}

impl ConvergenceConfig {
    /// High-resolution config for close zoom
    /// Phase 7: Used for zoom-based LOD switching
    #[allow(dead_code)] // Phase 7: Zoom LOD
    pub fn high_res() -> Self {
        Self {
            resolution: (256, 128),
            max_cells: 32768,
            ..Default::default()
        }
    }

    /// Low-resolution config for distant view
    /// Phase 7: Used for zoom-based LOD switching
    #[allow(dead_code)] // Phase 7: Zoom LOD
    pub fn low_res() -> Self {
        Self {
            resolution: (64, 32),
            max_cells: 2048,
            ..Default::default()
        }
    }
}

/// A single convergence cell with probability data
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ConvergenceCell {
    /// Geographic position (longitude, latitude in radians)
    pub position: Vec2,
    /// Probability intensity (0.0 - 1.0)
    pub intensity: f32,
    /// Rate of change (for animation)
    pub rate: f32,
    /// Temporal confidence (how stable is this prediction)
    pub confidence: f32,
    /// Vorticity contribution (for color mapping)
    pub vorticity: f32,
}

impl ConvergenceCell {
    /// Check if this cell exceeds visibility threshold
    #[inline]
    pub fn is_visible(&self, threshold: f32) -> bool {
        self.intensity > threshold
    }

    /// Check if this is a high-intensity zone
    /// Phase 8: Used for probability probe interaction
    #[inline]
    #[allow(dead_code)] // Phase 8: Probability probe
    pub fn is_high_intensity(&self, threshold: f32) -> bool {
        self.intensity > threshold
    }

    /// Convert geographic position to 3D globe position
    /// Phase 8: Used for probe anchor positioning
    #[allow(dead_code)] // Phase 8: Probe anchor
    pub fn to_globe_position(self, radius: f32) -> Vec3 {
        let lon = self.position.x;
        let lat = self.position.y;
        
        Vec3::new(
            radius * lat.cos() * lon.cos(),
            radius * lat.sin(),
            radius * lat.cos() * lon.sin(),
        )
    }
}

/// Convergence field containing probability data across the globe
#[derive(Debug)]
pub struct ConvergenceField {
    /// Configuration
    pub config: ConvergenceConfig,
    /// Grid of convergence cells (row-major: [lat][lon])
    cells: Vec<ConvergenceCell>,
    /// Frame index this data corresponds to
    pub frame_index: u64,
    /// Global maximum intensity (for normalization)
    pub max_intensity: f32,
    /// Number of high-intensity zones detected
    pub high_intensity_count: u32,
}

impl ConvergenceField {
    /// Create an empty convergence field
    pub fn new(config: ConvergenceConfig) -> Self {
        let cell_count = (config.resolution.0 * config.resolution.1) as usize;
        Self {
            config,
            cells: vec![ConvergenceCell::default(); cell_count],
            frame_index: 0,
            max_intensity: 0.0,
            high_intensity_count: 0,
        }
    }

    /// Generate synthetic convergence data for testing
    /// Creates realistic-looking storm convergence patterns
    pub fn generate_synthetic(&mut self, time: f32) {
        let (width, height) = self.config.resolution;
        self.max_intensity = 0.0;
        self.high_intensity_count = 0;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                
                // Geographic coordinates
                let lon = (x as f32 / width as f32) * std::f32::consts::TAU - std::f32::consts::PI;
                let lat = (y as f32 / height as f32) * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;

                // Create several convergence centers (simulating storm systems)
                let mut intensity = 0.0_f32;
                let mut vorticity = 0.0_f32;

                // Storm 1: North Atlantic hurricane
                let storm1_center = Vec2::new(-0.8 + time * 0.05, 0.4);
                let dist1 = ((lon - storm1_center.x).powi(2) + (lat - storm1_center.y).powi(2)).sqrt();
                let contrib1 = (-dist1 * 3.0).exp() * 0.9;
                intensity += contrib1;
                vorticity += contrib1 * 0.8 * (time * 2.0).sin();

                // Storm 2: Pacific typhoon
                let storm2_center = Vec2::new(2.5 + time * 0.03, 0.3);
                let dist2 = ((lon - storm2_center.x).powi(2) + (lat - storm2_center.y).powi(2)).sqrt();
                let contrib2 = (-dist2 * 4.0).exp() * 0.85;
                intensity += contrib2;
                vorticity -= contrib2 * 0.7 * (time * 1.5 + 1.0).sin();

                // Storm 3: Southern hemisphere low
                let storm3_center = Vec2::new(0.5, -0.5 + time * 0.02);
                let dist3 = ((lon - storm3_center.x).powi(2) + (lat - storm3_center.y).powi(2)).sqrt();
                let contrib3 = (-dist3 * 3.5).exp() * 0.7;
                intensity += contrib3;
                vorticity += contrib3 * 0.6 * (time * 2.5).cos();

                // Storm 4: Jet stream convergence zone
                let jet_lat = 0.7 + 0.2 * (lon * 2.0 + time * 0.1).sin();
                let jet_dist = (lat - jet_lat).abs();
                let jet_contrib = (-jet_dist * 5.0).exp() * 0.4 * (0.5 + 0.5 * (lon * 3.0).sin());
                intensity += jet_contrib;

                // Clamp and normalize
                intensity = intensity.clamp(0.0, 1.0);
                vorticity = vorticity.clamp(-1.0, 1.0);

                // Rate of change (derivative approximation)
                let rate = (time * 3.0 + lon * 2.0 + lat).sin() * intensity * 0.2;

                // Confidence decreases with intensity (more uncertain at extremes)
                let confidence = 1.0 - (intensity - 0.5).abs() * 0.6;

                self.cells[idx] = ConvergenceCell {
                    position: Vec2::new(lon, lat),
                    intensity,
                    rate,
                    confidence,
                    vorticity,
                };

                // Track statistics
                self.max_intensity = self.max_intensity.max(intensity);
                if intensity > self.config.high_intensity_threshold {
                    self.high_intensity_count += 1;
                }
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get cell at specific grid coordinates
    /// Phase 8: Used for probability probe cell lookup
    #[inline]
    #[allow(dead_code)] // Phase 8: Probability probe
    pub fn get_cell(&self, x: u32, y: u32) -> Option<&ConvergenceCell> {
        let idx = (y * self.config.resolution.0 + x) as usize;
        self.cells.get(idx)
    }

    /// Get cell at geographic coordinates (lon/lat in radians)
    /// Phase 8: Used for probability probe interpolated lookup
    #[allow(dead_code)] // Phase 8: Probability probe
    pub fn sample(&self, lon: f32, lat: f32) -> ConvergenceCell {
        let (width, height) = self.config.resolution;
        
        // Normalize to [0, 1] range
        let u = (lon + std::f32::consts::PI) / std::f32::consts::TAU;
        let v = (lat + std::f32::consts::FRAC_PI_2) / std::f32::consts::PI;

        // Grid coordinates
        let fx = u * (width - 1) as f32;
        let fy = v * (height - 1) as f32;
        
        let x0 = (fx.floor() as u32).min(width - 2);
        let y0 = (fy.floor() as u32).min(height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = fx.fract();
        let ty = fy.fract();

        // Bilinear interpolation
        let c00 = self.get_cell(x0, y0).copied().unwrap_or_default();
        let c10 = self.get_cell(x1, y0).copied().unwrap_or_default();
        let c01 = self.get_cell(x0, y1).copied().unwrap_or_default();
        let c11 = self.get_cell(x1, y1).copied().unwrap_or_default();

        ConvergenceCell {
            position: Vec2::new(lon, lat),
            intensity: bilinear(c00.intensity, c10.intensity, c01.intensity, c11.intensity, tx, ty),
            rate: bilinear(c00.rate, c10.rate, c01.rate, c11.rate, tx, ty),
            confidence: bilinear(c00.confidence, c10.confidence, c01.confidence, c11.confidence, tx, ty),
            vorticity: bilinear(c00.vorticity, c10.vorticity, c01.vorticity, c11.vorticity, tx, ty),
        }
    }

    /// Get all cells above visibility threshold (for rendering)
    pub fn visible_cells(&self) -> impl Iterator<Item = &ConvergenceCell> {
        self.cells.iter().filter(|c| c.is_visible(self.config.visibility_threshold))
    }

    /// Get all high-intensity cells (for interaction highlighting)
    /// Phase 8: Used for hover state detection
    #[allow(dead_code)] // Phase 8: Hover detection
    pub fn high_intensity_cells(&self) -> impl Iterator<Item = &ConvergenceCell> {
        self.cells.iter().filter(|c| c.is_high_intensity(self.config.high_intensity_threshold))
    }

    /// Get raw cell data for GPU upload
    /// Phase 8: Alternative upload path
    #[allow(dead_code)] // Phase 8: Alternative GPU upload
    pub fn cell_data(&self) -> &[ConvergenceCell] {
        &self.cells
    }

    /// Get cell count
    /// Phase 8: Telemetry display
    #[allow(dead_code)] // Phase 8: Telemetry
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

/// Bilinear interpolation helper
/// Phase 8: Used by sample() for probe lookup
#[inline]
#[allow(dead_code)] // Phase 8: Probe sampling
fn bilinear(c00: f32, c10: f32, c01: f32, c11: f32, tx: f32, ty: f32) -> f32 {
    let a = c00 * (1.0 - tx) + c10 * tx;
    let b = c01 * (1.0 - tx) + c11 * tx;
    a * (1.0 - ty) + b * ty
}

/// Bridge for reading convergence data from RAM bridge
/// Phase 6: Scaffolding for future RAM bridge integration
pub struct ConvergenceBridge {
    /// Current convergence field
    field: ConvergenceField,
    /// Shared memory path (future use)
    _shm_path: String,
}

impl ConvergenceBridge {
    /// Create a new convergence bridge
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            field: ConvergenceField::new(config),
            _shm_path: String::from("/dev/shm/sovereign_convergence"),
        }
    }

    /// Update from RAM bridge (currently generates synthetic data)
    /// Phase 7: RAM Bridge integration complete in main_phase7.rs
    /// This method provides synthetic fallback for standalone testing
    pub fn update(&mut self, time: f32) {
        // Synthetic data fallback - Phase 7 reads directly from bridge
        self.field.generate_synthetic(time);
    }

    /// Get current field reference
    pub fn field(&self) -> &ConvergenceField {
        &self.field
    }

    /// Get mutable field reference
    /// Phase 8: Used for scenario injection
    #[allow(dead_code)] // Phase 8: Scenario injection
    pub fn field_mut(&mut self) -> &mut ConvergenceField {
        &mut self.field
    }
}

/// GPU-ready convergence cell for shader upload
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuConvergenceCell {
    /// Position (lon, lat, intensity, vorticity)
    pub data: [f32; 4],
}

impl From<&ConvergenceCell> for GpuConvergenceCell {
    fn from(cell: &ConvergenceCell) -> Self {
        Self {
            data: [cell.position.x, cell.position.y, cell.intensity, cell.vorticity],
        }
    }
}

/// Convergence uniforms for shader
/// Note: Must match convergence.wgsl struct layout exactly (160 bytes)
/// WGSL alignment: vec2 requires 8-byte alignment, vec4 requires 16-byte alignment
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvergenceUniforms {
    /// View-projection matrix (64 bytes, offset 0-64)
    pub view_proj: [[f32; 4]; 4],
    /// Camera position for RTE transformation (16 bytes, offset 64-80) - w component unused
    pub camera_pos: [f32; 4],
    /// Globe radius (4 bytes, offset 80-84)
    pub globe_radius: f32,
    /// Time for animation (4 bytes, offset 84-88)
    pub time: f32,
    /// Visibility threshold (4 bytes, offset 88-92)
    pub visibility_threshold: f32,
    /// High intensity threshold (4 bytes, offset 92-96)
    pub high_intensity_threshold: f32,
    /// Pulse frequency (4 bytes, offset 96-100)
    pub pulse_frequency: f32,
    /// Max intensity for normalization (4 bytes, offset 100-104)
    pub max_intensity: f32,
    /// Phase 8: Appendix D - Hover position (8 bytes, offset 104-112)
    pub hover_pos: [f32; 2],
    /// Padding for alignment (4 bytes, offset 112-116)
    pub _padding_a: f32,
    /// Phase 8: Appendix D - Hover intensity (4 bytes, offset 116-120)
    pub hover_intensity: f32,
    /// Phase 8: Appendix D - Ghost mode (4 bytes, offset 120-124)
    pub ghost_mode: f32,
    /// Explicit padding for vec2 alignment (4 bytes, offset 124-128)
    pub _pad1: f32,
    /// Phase 8: Ghost mode selected node position (8 bytes, offset 128-136)
    pub ghost_selected_pos: [f32; 2],
    /// Explicit padding for vec4 alignment (8 bytes, offset 136-144)
    pub _pad2: [f32; 2],
    /// Final padding (16 bytes, offset 144-160)
    pub _padding: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_field_creation() {
        let config = ConvergenceConfig::default();
        let field = ConvergenceField::new(config);
        assert_eq!(field.cell_count(), 128 * 64);
    }

    #[test]
    fn test_synthetic_generation() {
        let config = ConvergenceConfig::default();
        let mut field = ConvergenceField::new(config);
        field.generate_synthetic(0.0);
        
        assert!(field.max_intensity > 0.0);
        assert!(field.max_intensity <= 1.0);
    }

    #[test]
    fn test_bilinear_sampling() {
        let config = ConvergenceConfig::default();
        let mut field = ConvergenceField::new(config);
        field.generate_synthetic(0.0);
        
        let sample = field.sample(0.0, 0.0);
        assert!(sample.intensity >= 0.0);
        assert!(sample.intensity <= 1.0);
    }

    #[test]
    fn test_visible_cells() {
        let config = ConvergenceConfig::default();
        let mut field = ConvergenceField::new(config);
        field.generate_synthetic(0.0);
        
        let visible_count = field.visible_cells().count();
        assert!(visible_count > 0);
        assert!(visible_count <= field.cell_count());
    }
}

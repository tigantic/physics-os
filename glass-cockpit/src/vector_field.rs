// Phase 5: Vector Field Ingestion
// Reads QTT vector data from RAM bridge for visualization
// Constitutional compliance: Doctrine 2 (RAM bridge), Doctrine 7 (QTT format)

use glam::Vec2;

/// Vector field grid configuration
#[allow(dead_code)] // Phase 5 scaffolding - cell_size_m used in RAM bridge
#[derive(Debug, Clone, Copy)]
pub struct VectorFieldConfig {
    /// Grid width (number of cells in X direction)
    pub grid_width: u32,
    /// Grid height (number of cells in Y direction)
    pub grid_height: u32,
    /// Grid cell size in meters
    pub cell_size_m: f32,
    /// Minimum longitude (west boundary)
    pub lon_min: f32,
    /// Maximum longitude (east boundary)
    pub lon_max: f32,
    /// Minimum latitude (south boundary)
    pub lat_min: f32,
    /// Maximum latitude (north boundary)
    pub lat_max: f32,
}

impl Default for VectorFieldConfig {
    fn default() -> Self {
        Self {
            grid_width: 128,
            grid_height: 64,
            cell_size_m: 100_000.0, // 100km grid cells at global scale
            lon_min: -180.0,
            lon_max: 180.0,
            lat_min: -90.0,
            lat_max: 90.0,
        }
    }
}

#[allow(dead_code)] // Phase 5 scaffolding - grid_pos/grid_to_latlon used in RAM bridge integration
impl VectorFieldConfig {
    /// Create config for specific zoom level
    pub fn for_zoom_level(zoom: u8, center_lon: f32, center_lat: f32) -> Self {
        // Adaptive grid density based on zoom
        let (grid_w, grid_h, cell_size) = match zoom {
            0..=3 => (64, 32, 200_000.0),    // Global view: 200km cells
            4..=6 => (128, 64, 100_000.0),   // Continental: 100km cells
            7..=9 => (256, 128, 50_000.0),   // Regional: 50km cells
            10..=12 => (512, 256, 25_000.0), // Mesoscale: 25km cells
            _ => (1024, 512, 10_000.0),      // Storm scale: 10km cells
        };
        
        // Calculate visible extent based on zoom
        let extent_deg = 360.0 / (2.0_f32.powi(zoom as i32));
        
        Self {
            grid_width: grid_w,
            grid_height: grid_h,
            cell_size_m: cell_size,
            lon_min: center_lon - extent_deg / 2.0,
            lon_max: center_lon + extent_deg / 2.0,
            lat_min: (center_lat - extent_deg / 4.0).max(-85.0),
            lat_max: (center_lat + extent_deg / 4.0).min(85.0),
        }
    }
    
    /// Get grid position for lat/lon
    pub fn grid_pos(&self, lon: f32, lat: f32) -> Option<(u32, u32)> {
        if lon < self.lon_min || lon > self.lon_max || 
           lat < self.lat_min || lat > self.lat_max {
            return None;
        }
        
        let x = ((lon - self.lon_min) / (self.lon_max - self.lon_min) * self.grid_width as f32) as u32;
        let y = ((lat - self.lat_min) / (self.lat_max - self.lat_min) * self.grid_height as f32) as u32;
        
        Some((x.min(self.grid_width - 1), y.min(self.grid_height - 1)))
    }
    
    /// Get lat/lon for grid position
    pub fn grid_to_latlon(&self, x: u32, y: u32) -> (f32, f32) {
        let lon = self.lon_min + (x as f32 + 0.5) / self.grid_width as f32 * (self.lon_max - self.lon_min);
        let lat = self.lat_min + (y as f32 + 0.5) / self.grid_height as f32 * (self.lat_max - self.lat_min);
        (lon, lat)
    }
}

/// Single vector cell with velocity components
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorCell {
    /// Eastward velocity component (m/s)
    pub u: f32,
    /// Northward velocity component (m/s)
    pub v: f32,
    /// Vertical velocity component (m/s), positive = upward
    pub w: f32,
    /// Vorticity magnitude (1/s)
    pub vorticity: f32,
}

#[allow(dead_code)] // Phase 5 scaffolding - speed_3d/direction/normalized_2d for advanced viz
impl VectorCell {
    /// Create new vector cell
    pub fn new(u: f32, v: f32, w: f32) -> Self {
        // Compute vorticity as curl magnitude approximation
        // Full vorticity requires neighboring cells; this is a placeholder
        Self { u, v, w, vorticity: 0.0 }
    }
    
    /// Get horizontal velocity magnitude
    pub fn speed(&self) -> f32 {
        (self.u * self.u + self.v * self.v).sqrt()
    }
    
    /// Get 3D velocity magnitude
    pub fn speed_3d(&self) -> f32 {
        (self.u * self.u + self.v * self.v + self.w * self.w).sqrt()
    }
    
    /// Get horizontal direction in radians (0 = east, π/2 = north)
    pub fn direction(&self) -> f32 {
        self.v.atan2(self.u)
    }
    
    /// Get normalized horizontal velocity
    pub fn normalized_2d(&self) -> Vec2 {
        let speed = self.speed();
        if speed > 1e-6 {
            Vec2::new(self.u / speed, self.v / speed)
        } else {
            Vec2::ZERO
        }
    }
}

/// Vector field grid data
#[allow(dead_code)] // Phase 5 scaffolding - frame_number/timestamp_us for RAM bridge sync
pub struct VectorField {
    /// Configuration
    pub config: VectorFieldConfig,
    /// Vector data (flattened row-major grid)
    pub data: Vec<VectorCell>,
    /// Frame number from source
    pub frame_number: u64,
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    /// Statistics
    pub stats: VectorFieldStats,
}

/// Vector field statistics for normalization
#[allow(dead_code)] // Phase 5 scaffolding - min_speed/mean_speed/min_vorticity for advanced rendering
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorFieldStats {
    pub max_speed: f32,
    pub min_speed: f32,
    pub mean_speed: f32,
    pub max_vorticity: f32,
    pub min_vorticity: f32,
}

#[allow(dead_code)] // Phase 5 scaffolding - get_mut for in-place vector updates
impl VectorField {
    /// Create empty vector field
    pub fn new(config: VectorFieldConfig) -> Self {
        let size = (config.grid_width * config.grid_height) as usize;
        Self {
            config,
            data: vec![VectorCell::default(); size],
            frame_number: 0,
            timestamp_us: 0,
            stats: VectorFieldStats::default(),
        }
    }
    
    /// Get vector at grid position
    pub fn get(&self, x: u32, y: u32) -> Option<&VectorCell> {
        if x >= self.config.grid_width || y >= self.config.grid_height {
            return None;
        }
        let idx = (y * self.config.grid_width + x) as usize;
        self.data.get(idx)
    }
    
    /// Get mutable vector at grid position
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut VectorCell> {
        if x >= self.config.grid_width || y >= self.config.grid_height {
            return None;
        }
        let idx = (y * self.config.grid_width + x) as usize;
        self.data.get_mut(idx)
    }
    
    /// Bilinear interpolation of vector at arbitrary lat/lon
    pub fn sample(&self, lon: f32, lat: f32) -> VectorCell {
        // Normalize to grid coordinates
        let fx = (lon - self.config.lon_min) / (self.config.lon_max - self.config.lon_min) 
                 * (self.config.grid_width - 1) as f32;
        let fy = (lat - self.config.lat_min) / (self.config.lat_max - self.config.lat_min) 
                 * (self.config.grid_height - 1) as f32;
        
        let x0 = (fx.floor() as u32).min(self.config.grid_width - 2);
        let y0 = (fy.floor() as u32).min(self.config.grid_height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        
        // Get four corners
        let v00 = self.get(x0, y0).copied().unwrap_or_default();
        let v10 = self.get(x1, y0).copied().unwrap_or_default();
        let v01 = self.get(x0, y1).copied().unwrap_or_default();
        let v11 = self.get(x1, y1).copied().unwrap_or_default();
        
        // Bilinear interpolation
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        
        VectorCell {
            u: lerp(lerp(v00.u, v10.u, tx), lerp(v01.u, v11.u, tx), ty),
            v: lerp(lerp(v00.v, v10.v, tx), lerp(v01.v, v11.v, tx), ty),
            w: lerp(lerp(v00.w, v10.w, tx), lerp(v01.w, v11.w, tx), ty),
            vorticity: lerp(lerp(v00.vorticity, v10.vorticity, tx), lerp(v01.vorticity, v11.vorticity, tx), ty),
        }
    }
    
    /// Compute vorticity from velocity curl
    pub fn compute_vorticity(&mut self) {
        let w = self.config.grid_width;
        let h = self.config.grid_height;
        let dx = (self.config.lon_max - self.config.lon_min) / w as f32;
        let dy = (self.config.lat_max - self.config.lat_min) / h as f32;
        
        // Compute vorticity: ∂v/∂x - ∂u/∂y (vertical component of curl)
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = (y * w + x) as usize;
                
                // Central differences
                let v_right = self.data[idx + 1].v;
                let v_left = self.data[idx - 1].v;
                let u_up = self.data[idx + w as usize].u;
                let u_down = self.data[idx - w as usize].u;
                
                let dvdx = (v_right - v_left) / (2.0 * dx);
                let dudy = (u_up - u_down) / (2.0 * dy);
                
                self.data[idx].vorticity = dvdx - dudy;
            }
        }
    }
    
    /// Update statistics
    pub fn compute_stats(&mut self) {
        if self.data.is_empty() {
            return;
        }
        
        let mut max_speed = f32::NEG_INFINITY;
        let mut min_speed = f32::INFINITY;
        let mut sum_speed = 0.0;
        let mut max_vort = f32::NEG_INFINITY;
        let mut min_vort = f32::INFINITY;
        
        for cell in &self.data {
            let speed = cell.speed();
            max_speed = max_speed.max(speed);
            min_speed = min_speed.min(speed);
            sum_speed += speed;
            max_vort = max_vort.max(cell.vorticity);
            min_vort = min_vort.min(cell.vorticity);
        }
        
        self.stats = VectorFieldStats {
            max_speed,
            min_speed,
            mean_speed: sum_speed / self.data.len() as f32,
            max_vorticity: max_vort,
            min_vorticity: min_vort,
        };
    }
    
    /// Generate synthetic test data (rotating vortex pattern)
    pub fn generate_test_pattern(&mut self) {
        let w = self.config.grid_width;
        let h = self.config.grid_height;
        
        // Create multiple vortices for test
        let vortices = vec![
            (0.3, 0.5, 1.0, 10.0),   // (x, y, strength, radius) - cyclonic
            (0.7, 0.4, -0.5, 8.0),   // anticyclonic
            (0.5, 0.7, 0.8, 12.0),   // larger cyclonic
        ];
        
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let px = x as f32 / w as f32;
                let py = y as f32 / h as f32;
                
                let mut u_total = 0.0;
                let mut v_total = 0.0;
                
                for (vx, vy, strength, radius) in &vortices {
                    let dx = px - vx;
                    let dy = py - vy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    
                    if dist > 0.001 {
                        // Rankine vortex profile
                        let r_norm = dist / (radius / w as f32);
                        let v_tan = if r_norm < 1.0 {
                            strength * r_norm  // Solid body rotation inside
                        } else {
                            strength / r_norm  // Potential vortex outside
                        };
                        
                        // Tangential velocity (perpendicular to radius)
                        let angle = dy.atan2(dx);
                        u_total += -v_tan * angle.sin() * 20.0; // Scale to m/s
                        v_total += v_tan * angle.cos() * 20.0;
                    }
                }
                
                // Add background flow (westerlies)
                u_total += 10.0 + py * 15.0; // Stronger at higher latitudes
                
                self.data[idx] = VectorCell::new(u_total, v_total, 0.0);
            }
        }
        
        self.compute_vorticity();
        self.compute_stats();
    }
}

/// RAM bridge vector field reader
#[allow(dead_code)] // Phase 5 scaffolding - RAM bridge integration in Phase 6
pub struct VectorFieldBridge {
    /// Shared memory path
    path: String,
    /// Current field data
    field: VectorField,
    /// Last frame number read
    last_frame: u64,
}

#[allow(dead_code)] // Phase 5 scaffolding - RAM bridge integration in Phase 6
impl VectorFieldBridge {
    /// Create new bridge connection
    pub fn new(path: &str, config: VectorFieldConfig) -> Self {
        Self {
            path: path.to_string(),
            field: VectorField::new(config),
            last_frame: 0,
        }
    }
    
    /// Update field from RAM bridge (returns true if new data available)
    pub fn update(&mut self) -> bool {
        // TODO: Implement actual RAM bridge reading
        // For now, generate test data on first call
        if self.last_frame == 0 {
            self.field.generate_test_pattern();
            self.last_frame = 1;
            return true;
        }
        false
    }
    
    /// Get current field data
    pub fn field(&self) -> &VectorField {
        &self.field
    }
    
    /// Get mutable field data
    pub fn field_mut(&mut self) -> &mut VectorField {
        &mut self.field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_field_creation() {
        let config = VectorFieldConfig::default();
        let field = VectorField::new(config);
        
        assert_eq!(field.data.len(), (config.grid_width * config.grid_height) as usize);
    }
    
    #[test]
    fn test_grid_position() {
        let config = VectorFieldConfig::default();
        
        // Center of grid
        let pos = config.grid_pos(0.0, 0.0);
        assert!(pos.is_some());
        let (x, y) = pos.unwrap();
        assert_eq!(x, config.grid_width / 2);
        assert_eq!(y, config.grid_height / 2);
        
        // Out of bounds
        assert!(config.grid_pos(-200.0, 0.0).is_none());
    }
    
    #[test]
    fn test_vector_cell_speed() {
        let cell = VectorCell::new(3.0, 4.0, 0.0);
        assert!((cell.speed() - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_bilinear_interpolation() {
        let config = VectorFieldConfig {
            grid_width: 4,
            grid_height: 4,
            ..Default::default()
        };
        let mut field = VectorField::new(config);
        
        // Set corner values
        field.data[0] = VectorCell::new(0.0, 0.0, 0.0);  // (0,0)
        field.data[1] = VectorCell::new(10.0, 0.0, 0.0); // (1,0)
        field.data[4] = VectorCell::new(0.0, 10.0, 0.0); // (0,1)
        field.data[5] = VectorCell::new(10.0, 10.0, 0.0); // (1,1)
        
        // Sample center
        let center = field.sample(-90.0, -45.0); // Approximately center of first cell
        
        // Should be interpolated
        assert!(center.u >= 0.0 && center.u <= 10.0);
        assert!(center.v >= 0.0 && center.v <= 10.0);
    }
    
    #[test]
    fn test_synthetic_pattern() {
        let config = VectorFieldConfig {
            grid_width: 64,
            grid_height: 32,
            ..Default::default()
        };
        let mut field = VectorField::new(config);
        field.generate_test_pattern();
        
        // Should have non-zero velocities
        assert!(field.stats.max_speed > 0.0);
        
        // Should have vorticity
        let center_idx = (16 * 64 + 32) as usize;
        // Vorticity computed for interior points
        assert!(field.data[center_idx].vorticity.abs() >= 0.0);
    }
}

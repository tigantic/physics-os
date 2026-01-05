// Phase 1: Weather Tensor Grid
// Multi-channel tensor extending VectorField with temperature, pressure, humidity
// Constitutional compliance: Doctrine 7 (QTT format), Doctrine 3 (GPU-first)
#![allow(dead_code)]  // Phase 1 infrastructure - ready for NOAA integration

use crate::vector_field::{VectorField, VectorFieldConfig, VectorCell, VectorFieldStats};
use crate::noaa_fetcher::ForecastModel;

/// Extended weather data for a single grid cell
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WeatherCell {
    // === Wind Components (from VectorCell) ===
    /// Eastward velocity (m/s)
    pub u: f32,
    /// Northward velocity (m/s)
    pub v: f32,
    /// Vertical velocity (m/s, positive = upward)
    pub w: f32,
    /// Vorticity magnitude (1/s)
    pub vorticity: f32,
    
    // === Thermodynamic Variables ===
    /// Temperature (Kelvin)
    pub temperature: f32,
    /// Relative humidity (0.0-1.0)
    pub humidity: f32,
    /// Pressure (hPa)
    pub pressure: f32,
    /// Geopotential height (m)
    pub geopotential: f32,
    
    // === Derived Fields (computed) ===
    /// Divergence (∂u/∂x + ∂v/∂y)
    pub divergence: f32,
    /// Convergence zone flag (0-1)
    pub convergence: f32,
    /// Cloud potential (derived from RH + vertical motion)
    pub cloud_potential: f32,
    /// Padding for GPU alignment (16-byte aligned)
    pub _pad: f32,
}

impl WeatherCell {
    /// Create from basic wind data
    pub fn from_wind(u: f32, v: f32, w: f32) -> Self {
        Self {
            u, v, w,
            ..Default::default()
        }
    }
    
    /// Convert to VectorCell (for particle advection)
    pub fn to_vector_cell(self) -> VectorCell {
        VectorCell {
            u: self.u,
            v: self.v,
            w: self.w,
            vorticity: self.vorticity,
        }
    }
    
    /// Get wind speed (horizontal)
    pub fn wind_speed(&self) -> f32 {
        (self.u * self.u + self.v * self.v).sqrt()
    }
    
    /// Get wind direction in degrees (meteorological convention: direction FROM)
    pub fn wind_direction(&self) -> f32 {
        let dir = self.v.atan2(self.u).to_degrees();
        // Convert from math convention to meteorological
        (270.0 - dir).rem_euclid(360.0)
    }
    
    /// Get temperature in Celsius
    pub fn temp_celsius(&self) -> f32 {
        self.temperature - 273.15
    }
    
    /// Get temperature in Fahrenheit
    pub fn temp_fahrenheit(&self) -> f32 {
        self.temp_celsius() * 1.8 + 32.0
    }
    
    /// Compute derived fields from neighbors
    pub fn compute_derived(&mut self, neighbors: &WeatherNeighbors, dx: f32, dy: f32) {
        // Divergence: ∂u/∂x + ∂v/∂y (central differences)
        let dudx = (neighbors.east.u - neighbors.west.u) / (2.0 * dx);
        let dvdy = (neighbors.north.v - neighbors.south.v) / (2.0 * dy);
        self.divergence = dudx + dvdy;
        
        // Vorticity: ∂v/∂x - ∂u/∂y
        let dvdx = (neighbors.east.v - neighbors.west.v) / (2.0 * dx);
        let dudy = (neighbors.north.u - neighbors.south.u) / (2.0 * dy);
        self.vorticity = dvdx - dudy;
        
        // Convergence zone (negative divergence = convergence)
        self.convergence = (-self.divergence).clamp(0.0, 1.0) / 0.0001; // Normalize
        
        // Cloud potential: high RH + upward motion
        let rh_factor = (self.humidity - 0.7).max(0.0) / 0.3;
        let w_factor = self.w.max(0.0) / 0.5;
        self.cloud_potential = (rh_factor * 0.7 + w_factor * 0.3).min(1.0);
    }
}

/// Neighbor cells for computing spatial derivatives
pub struct WeatherNeighbors {
    pub north: WeatherCell,
    pub south: WeatherCell,
    pub east: WeatherCell,
    pub west: WeatherCell,
}

/// Statistics for weather tensor
#[derive(Debug, Clone, Copy, Default)]
pub struct WeatherTensorStats {
    // Wind stats
    pub max_wind_speed: f32,
    pub mean_wind_speed: f32,
    pub max_vorticity: f32,
    
    // Temperature stats
    pub max_temperature: f32,
    pub min_temperature: f32,
    pub mean_temperature: f32,
    
    // Moisture stats
    pub max_humidity: f32,
    pub mean_humidity: f32,
    
    // Pressure stats
    pub max_pressure: f32,
    pub min_pressure: f32,
}

/// Multi-channel weather tensor grid
pub struct WeatherTensor {
    /// Configuration (inherited from VectorField)
    pub config: VectorFieldConfig,
    /// Weather data (flattened row-major grid)
    pub data: Vec<WeatherCell>,
    /// Source model
    pub model: ForecastModel,
    /// Pressure level (hPa)
    pub level_hpa: u32,
    /// Valid time (when this data is valid for)
    pub valid_time_utc: String,
    /// Statistics
    pub stats: WeatherTensorStats,
    /// Data completeness (0.0-1.0)
    pub completeness: f32,
}

impl WeatherTensor {
    /// Create empty weather tensor
    pub fn new(config: VectorFieldConfig, model: ForecastModel, level_hpa: u32) -> Self {
        let size = (config.grid_width * config.grid_height) as usize;
        Self {
            config,
            data: vec![WeatherCell::default(); size],
            model,
            level_hpa,
            valid_time_utc: String::new(),
            stats: WeatherTensorStats::default(),
            completeness: 0.0,
        }
    }
    
    /// Create from decoded GRIB2 data
    pub fn from_grib_layers(
        config: VectorFieldConfig,
        model: ForecastModel,
        level_hpa: u32,
        u_wind: Option<&[f32]>,
        v_wind: Option<&[f32]>,
        temperature: Option<&[f32]>,
        humidity: Option<&[f32]>,
    ) -> Self {
        let size = (config.grid_width * config.grid_height) as usize;
        let mut tensor = Self::new(config, model, level_hpa);
        
        let mut channels_loaded = 0;
        
        // Load U-wind
        if let Some(u_data) = u_wind {
            if u_data.len() == size {
                for (i, &u) in u_data.iter().enumerate() {
                    tensor.data[i].u = u;
                }
                channels_loaded += 1;
            }
        }
        
        // Load V-wind
        if let Some(v_data) = v_wind {
            if v_data.len() == size {
                for (i, &v) in v_data.iter().enumerate() {
                    tensor.data[i].v = v;
                }
                channels_loaded += 1;
            }
        }
        
        // Load temperature
        if let Some(t_data) = temperature {
            if t_data.len() == size {
                for (i, &t) in t_data.iter().enumerate() {
                    tensor.data[i].temperature = t;
                }
                channels_loaded += 1;
            }
        }
        
        // Load humidity
        if let Some(rh_data) = humidity {
            if rh_data.len() == size {
                for (i, &rh) in rh_data.iter().enumerate() {
                    tensor.data[i].humidity = rh / 100.0; // Convert % to 0-1
                }
                channels_loaded += 1;
            }
        }
        
        tensor.completeness = channels_loaded as f32 / 4.0;
        tensor.compute_statistics();
        tensor.compute_derived_fields();
        
        tensor
    }
    
    /// Get cell at grid position
    pub fn get(&self, x: u32, y: u32) -> Option<&WeatherCell> {
        if x >= self.config.grid_width || y >= self.config.grid_height {
            return None;
        }
        let idx = (y * self.config.grid_width + x) as usize;
        self.data.get(idx)
    }
    
    /// Get mutable cell at grid position
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut WeatherCell> {
        if x >= self.config.grid_width || y >= self.config.grid_height {
            return None;
        }
        let idx = (y * self.config.grid_width + x) as usize;
        self.data.get_mut(idx)
    }
    
    /// Bilinear interpolation at arbitrary lat/lon
    pub fn sample(&self, lon: f32, lat: f32) -> WeatherCell {
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
        
        let c00 = self.get(x0, y0).copied().unwrap_or_default();
        let c10 = self.get(x1, y0).copied().unwrap_or_default();
        let c01 = self.get(x0, y1).copied().unwrap_or_default();
        let c11 = self.get(x1, y1).copied().unwrap_or_default();
        
        // Bilinear interpolation for each field
        WeatherCell {
            u: lerp_2d(c00.u, c10.u, c01.u, c11.u, tx, ty),
            v: lerp_2d(c00.v, c10.v, c01.v, c11.v, tx, ty),
            w: lerp_2d(c00.w, c10.w, c01.w, c11.w, tx, ty),
            vorticity: lerp_2d(c00.vorticity, c10.vorticity, c01.vorticity, c11.vorticity, tx, ty),
            temperature: lerp_2d(c00.temperature, c10.temperature, c01.temperature, c11.temperature, tx, ty),
            humidity: lerp_2d(c00.humidity, c10.humidity, c01.humidity, c11.humidity, tx, ty),
            pressure: lerp_2d(c00.pressure, c10.pressure, c01.pressure, c11.pressure, tx, ty),
            geopotential: lerp_2d(c00.geopotential, c10.geopotential, c01.geopotential, c11.geopotential, tx, ty),
            divergence: lerp_2d(c00.divergence, c10.divergence, c01.divergence, c11.divergence, tx, ty),
            convergence: lerp_2d(c00.convergence, c10.convergence, c01.convergence, c11.convergence, tx, ty),
            cloud_potential: lerp_2d(c00.cloud_potential, c10.cloud_potential, c01.cloud_potential, c11.cloud_potential, tx, ty),
            _pad: 0.0,
        }
    }
    
    /// Convert to VectorField for particle advection
    pub fn to_vector_field(&self) -> VectorField {
        let mut vf = VectorField::new(self.config);
        
        for (i, cell) in self.data.iter().enumerate() {
            vf.data[i] = cell.to_vector_cell();
        }
        
        vf.stats = VectorFieldStats {
            max_speed: self.stats.max_wind_speed,
            min_speed: 0.0,
            mean_speed: self.stats.mean_wind_speed,
            max_vorticity: self.stats.max_vorticity,
            min_vorticity: 0.0,
        };
        
        vf
    }
    
    /// Compute statistics from data
    pub fn compute_statistics(&mut self) {
        if self.data.is_empty() {
            return;
        }
        
        let mut max_speed = 0.0f32;
        let mut sum_speed = 0.0f32;
        let mut max_vort = 0.0f32;
        let mut max_temp = f32::MIN;
        let mut min_temp = f32::MAX;
        let mut sum_temp = 0.0f32;
        let mut max_rh = 0.0f32;
        let mut sum_rh = 0.0f32;
        let mut max_pres = 0.0f32;
        let mut min_pres = f32::MAX;
        
        for cell in &self.data {
            let speed = cell.wind_speed();
            max_speed = max_speed.max(speed);
            sum_speed += speed;
            max_vort = max_vort.max(cell.vorticity.abs());
            
            if cell.temperature > 0.0 {
                max_temp = max_temp.max(cell.temperature);
                min_temp = min_temp.min(cell.temperature);
                sum_temp += cell.temperature;
            }
            
            max_rh = max_rh.max(cell.humidity);
            sum_rh += cell.humidity;
            
            if cell.pressure > 0.0 {
                max_pres = max_pres.max(cell.pressure);
                min_pres = min_pres.min(cell.pressure);
            }
        }
        
        let n = self.data.len() as f32;
        self.stats = WeatherTensorStats {
            max_wind_speed: max_speed,
            mean_wind_speed: sum_speed / n,
            max_vorticity: max_vort,
            max_temperature: max_temp,
            min_temperature: if min_temp == f32::MAX { 0.0 } else { min_temp },
            mean_temperature: sum_temp / n,
            max_humidity: max_rh,
            mean_humidity: sum_rh / n,
            max_pressure: max_pres,
            min_pressure: if min_pres == f32::MAX { 0.0 } else { min_pres },
        };
    }
    
    /// Compute derived fields (vorticity, divergence, etc.)
    pub fn compute_derived_fields(&mut self) {
        let w = self.config.grid_width;
        let h = self.config.grid_height;
        
        // Grid spacing in degrees
        let dx = (self.config.lon_max - self.config.lon_min) / w as f32;
        let dy = (self.config.lat_max - self.config.lat_min) / h as f32;
        
        // Convert to approximate meters (at equator)
        let dx_m = dx * 111_000.0;
        let dy_m = dy * 111_000.0;
        
        // Clone data for neighbor access
        let data_copy = self.data.clone();
        
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = (y * w + x) as usize;
                
                let neighbors = WeatherNeighbors {
                    north: data_copy[((y + 1) * w + x) as usize],
                    south: data_copy[((y - 1) * w + x) as usize],
                    east: data_copy[(y * w + x + 1) as usize],
                    west: data_copy[(y * w + x - 1) as usize],
                };
                
                self.data[idx].compute_derived(&neighbors, dx_m, dy_m);
            }
        }
    }
    
    /// Get a channel as a flat f32 slice (for GPU upload)
    pub fn channel_data(&self, channel: WeatherChannel) -> Vec<f32> {
        self.data.iter().map(|cell| {
            match channel {
                WeatherChannel::UWind => cell.u,
                WeatherChannel::VWind => cell.v,
                WeatherChannel::WVelocity => cell.w,
                WeatherChannel::Vorticity => cell.vorticity,
                WeatherChannel::Temperature => cell.temperature,
                WeatherChannel::Humidity => cell.humidity,
                WeatherChannel::Pressure => cell.pressure,
                WeatherChannel::Divergence => cell.divergence,
                WeatherChannel::Convergence => cell.convergence,
            }
        }).collect()
    }
}

/// Weather channel selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherChannel {
    UWind,
    VWind,
    WVelocity,
    Vorticity,
    Temperature,
    Humidity,
    Pressure,
    Divergence,
    Convergence,
}

/// Bilinear interpolation helper
fn lerp_2d(c00: f32, c10: f32, c01: f32, c11: f32, tx: f32, ty: f32) -> f32 {
    let a = c00 * (1.0 - tx) + c10 * tx;
    let b = c01 * (1.0 - tx) + c11 * tx;
    a * (1.0 - ty) + b * ty
}

/// LOD manager for multi-resolution weather data
pub struct WeatherLodManager {
    /// GFS tensor (global, 28km)
    pub gfs: Option<WeatherTensor>,
    /// HRRR tensor (CONUS, 3km)
    pub hrrr: Option<WeatherTensor>,
    /// Current zoom level
    zoom_level: u8,
    /// Center lon/lat
    center: (f32, f32),
}

impl WeatherLodManager {
    /// Create empty LOD manager
    pub fn new() -> Self {
        Self {
            gfs: None,
            hrrr: None,
            zoom_level: 0,
            center: (0.0, 0.0),
        }
    }
    
    /// Update view parameters
    pub fn update_view(&mut self, zoom: u8, center_lon: f32, center_lat: f32) {
        self.zoom_level = zoom;
        self.center = (center_lon, center_lat);
    }
    
    /// Get best available tensor for current view
    pub fn best_tensor(&self) -> Option<&WeatherTensor> {
        // Use HRRR for high zoom in CONUS
        if self.zoom_level >= 7 {
            if let Some(ref hrrr) = self.hrrr {
                if hrrr.model.covers(self.center.1, self.center.0) {
                    return Some(hrrr);
                }
            }
        }
        
        // Fall back to GFS
        self.gfs.as_ref()
    }
    
    /// Sample weather at lat/lon using best available data
    pub fn sample(&self, lon: f32, lat: f32) -> WeatherCell {
        if let Some(tensor) = self.best_tensor() {
            tensor.sample(lon, lat)
        } else {
            WeatherCell::default()
        }
    }
    
    /// Get model currently in use
    pub fn current_model(&self) -> Option<ForecastModel> {
        self.best_tensor().map(|t| t.model)
    }
}

impl Default for WeatherLodManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weather_cell_conversion() {
        let cell = WeatherCell {
            u: 10.0,
            v: 5.0,
            w: 0.5,
            temperature: 288.15, // 15°C
            humidity: 0.75,
            ..Default::default()
        };
        
        assert!((cell.wind_speed() - 11.18).abs() < 0.1);
        assert!((cell.temp_celsius() - 15.0).abs() < 0.01);
    }
    
    #[test]
    fn test_tensor_sampling() {
        let config = VectorFieldConfig::default();
        let mut tensor = WeatherTensor::new(config, ForecastModel::Gfs, 850);
        
        // Set a known value
        if let Some(cell) = tensor.get_mut(64, 32) {
            cell.u = 15.0;
            cell.v = 10.0;
            cell.temperature = 290.0;
        }
        
        // Sample near that point
        let sample = tensor.sample(0.0, 0.0); // Center of grid
        assert!(sample.u > 0.0 || sample.v > 0.0 || sample.temperature > 0.0);
    }
}

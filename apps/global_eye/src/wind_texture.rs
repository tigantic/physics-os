//! Wind Texture Management - Global Eye Phase 1C-8
//!
//! Creates and updates the Rgba32Float texture for wind visualization.

use wgpu::{Device, Queue, Texture, TextureView, Sampler};

/// Wind texture wrapper with associated resources.
pub struct WindTexture {
    /// The GPU texture (Rgba32Float)
    pub texture: Texture,
    /// Texture view for shader binding
    pub view: TextureView,
    /// Sampler for texture filtering
    pub sampler: Sampler,
    /// Current dimensions
    pub width: u32,
    pub height: u32,
}

impl WindTexture {
    /// Create a new wind texture with given dimensions.
    ///
    /// Exit Gate 1C-8: Must compile without wgpu validation errors.
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        // Create the texture
        // CRITICAL: Rgba32Float allows negative values (important for wind direction)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Global Wind Field"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Rgba32Float is CRITICAL:
            // - R = U-wind (can be negative for westward)
            // - G = V-wind (can be negative for southward)
            // - B = Magnitude (pre-computed)
            // - A = Reserved (can store temperature, pressure, etc.)
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Wind Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        Self {
            texture,
            view,
            sampler,
            width,
            height,
        }
    }
    
    /// Update the texture with new wind data.
    ///
    /// # Arguments
    /// * `queue` - WGPU command queue
    /// * `u_field` - East/West wind component (m/s)
    /// * `v_field` - North/South wind component (m/s)
    ///
    /// # Panics
    /// If field dimensions don't match texture size.
    pub fn update(&self, queue: &Queue, u_field: &[f32], v_field: &[f32]) {
        let pixel_count = (self.width * self.height) as usize;
        
        assert_eq!(u_field.len(), pixel_count, "U-field size mismatch");
        assert_eq!(v_field.len(), pixel_count, "V-field size mismatch");
        
        // Build RGBA32F buffer: [u, v, magnitude, 0]
        let mut rgba_data: Vec<f32> = Vec::with_capacity(pixel_count * 4);
        
        for i in 0..pixel_count {
            let u = u_field[i];
            let v = v_field[i];
            let magnitude = (u * u + v * v).sqrt();
            
            rgba_data.push(u);          // R = U-wind
            rgba_data.push(v);          // G = V-wind
            rgba_data.push(magnitude);  // B = Speed
            rgba_data.push(0.0);        // A = Reserved
        }
        
        // Upload to GPU
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4 * 4), // 4 channels * 4 bytes per f32
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Update from a WeatherFrame (zero-copy from shared memory).
    #[cfg(feature = "weather-bridge")]
    pub fn update_from_frame(&self, queue: &Queue, frame: &hyper_bridge::WeatherFrame) {
        self.update(queue, frame.u_field(), frame.v_field());
    }
    
    /// Create a synthetic test pattern for validation.
    pub fn fill_test_pattern(&self, queue: &Queue) {
        let pixel_count = (self.width * self.height) as usize;
        let mut rgba_data: Vec<f32> = Vec::with_capacity(pixel_count * 4);
        
        for y in 0..self.height {
            for x in 0..self.width {
                // Create a circular wind pattern (vortex)
                let cx = self.width as f32 / 2.0;
                let cy = self.height as f32 / 2.0;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                
                // Tangential wind (counterclockwise rotation)
                let u = -dy / dist * 20.0;
                let v = dx / dist * 20.0;
                let magnitude = (u * u + v * v).sqrt();
                
                rgba_data.push(u);
                rgba_data.push(v);
                rgba_data.push(magnitude);
                rgba_data.push(0.0);
            }
        }
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4 * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Get the bind group layout for this texture.
    pub fn bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wind Texture Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
    
    /// Create a bind group for this texture.
    pub fn bind_group(&self, device: &Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wind Texture Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_texture_size_calculation() {
        // HRRR CONUS is approximately 1799x1059
        let width = 1799u32;
        let height = 1059u32;
        let bytes_per_pixel = 4 * 4; // RGBA32F = 16 bytes per pixel
        let total_bytes = (width * height) as usize * bytes_per_pixel;
        
        // Should be about 30.5 MB
        assert!(total_bytes < 35_000_000, "Texture too large: {} bytes", total_bytes);
    }
}

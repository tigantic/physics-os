// Phase 4: Satellite Texture Manager
// Manages GPU textures for NASA GIBS satellite tiles
// Constitutional compliance: Doctrine 1 (GPU-first), Doctrine 3 (zero-copy where possible)
#![allow(dead_code)] // Texture manager infrastructure for Phase 3

use crate::tile_fetcher::{TileFetcher, TileCoord, TileStatus, GibsConfig};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Manages satellite imagery textures for globe rendering
pub struct SatelliteTextureManager {
    /// Tile fetcher for async tile loading
    fetcher: TileFetcher,
    /// GPU textures per tile coordinate
    tile_textures: HashMap<TileCoord, wgpu::Texture>,
    /// Composite texture atlas (equirectangular projection)
    pub atlas_texture: wgpu::Texture,
    pub atlas_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Atlas dimensions
    atlas_width: u32,
    atlas_height: u32,
    /// Current zoom level for tile requests
    current_zoom: u8,
    /// Tiles loaded into atlas
    loaded_tiles: HashMap<TileCoord, (u32, u32)>, // coord -> atlas position
}

impl SatelliteTextureManager {
    /// Create new satellite texture manager
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<Self> {
        // Create tile fetcher with default NASA GIBS config
        let fetcher = TileFetcher::new(GibsConfig::default())?;
        
        // Create atlas texture (2048x1024 equirectangular)
        let atlas_width = 2048;
        let atlas_height = 1024;
        
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Satellite Atlas Texture"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Satellite Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group layout for satellite textures
        // Must match globe.wgsl group(1) bindings:
        // binding 0: MaterialUniforms (uniform buffer)
        // binding 1: texture_2d_array (satellite_textures) 
        // binding 2: sampler
        // binding 3: texture_2d (fallback single texture)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Satellite Texture Bind Group Layout"),
            entries: &[
                // binding 0: MaterialUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_2d_array (satellite_textures)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: texture_2d (fallback single texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        
        // Create texture array for streaming tiles (256x256 × 128 layers)
        let array_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Satellite Texture Array"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 128,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let array_view = array_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        
        // Create material uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MaterialUniforms {
            base_color: [f32; 4],
            grid_color: [f32; 4],
            rim_color: [f32; 4],
            atmosphere_color: [f32; 4],
            atmosphere_density: f32,
            grid_thickness: f32,
            latitude_spacing: f32,
            longitude_spacing: f32,
        }
        
        let material_data = MaterialUniforms {
            base_color: [0.1, 0.2, 0.4, 1.0],
            grid_color: [0.0, 0.8, 1.0, 0.5],
            rim_color: [0.3, 0.6, 1.0, 1.0],
            atmosphere_color: [0.4, 0.7, 1.0, 0.3],
            atmosphere_density: 0.5,
            grid_thickness: 0.002,
            latitude_spacing: 10.0,
            longitude_spacing: 10.0,
        };
        
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Uniform Buffer"),
            contents: bytemuck::bytes_of(&material_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Satellite Texture Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
            ],
        });
        
        // Initialize with a visible blue sphere (bright enough to survive 0.2x shader tint)
        // 50*0.2 = 10, 100*0.2 = 20, 180*0.2 = 36 - visible dark blue
        let default_color = [50u8, 100, 180, 255]; // Bright blue ocean placeholder
        let default_data: Vec<u8> = (0..atlas_width * atlas_height)
            .flat_map(|_| default_color.iter().copied())
            .collect();
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &default_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_width * 4),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
        );
        
        Ok(Self {
            fetcher,
            tile_textures: HashMap::new(),
            atlas_texture,
            atlas_view,
            sampler,
            bind_group,
            bind_group_layout,
            atlas_width,
            atlas_height,
            current_zoom: 3, // Start at low zoom for global view
            loaded_tiles: HashMap::new(),
        })
    }
    
    /// Request tiles for the current camera view
    pub fn request_tiles_for_view(&mut self, camera_lat: f64, camera_lon: f64, zoom: f32) {
        // Map camera zoom to tile zoom level (0-9 for GIBS)
        let tile_zoom = ((1.0 / zoom) * 2.0).log2().clamp(0.0, 8.0) as u8;
        
        // Only request if zoom changed significantly
        if tile_zoom != self.current_zoom {
            println!("🌍 Requesting tiles at zoom {} (camera lat={:.2}, lon={:.2})", tile_zoom, camera_lat, camera_lon);
            self.current_zoom = tile_zoom;
        }
        
        // Request visible tiles
        self.fetcher.request_visible_tiles(camera_lat, camera_lon, self.current_zoom);
    }
    
    /// Poll for fetched tiles and upload to GPU
    pub fn update(&mut self, queue: &wgpu::Queue) {
        let results = self.fetcher.poll_results();
        
        if !results.is_empty() {
            println!("📡 Received {} tile results", results.len());
        }
        
        for (coord, status) in results {
            match &status {
                TileStatus::Ready(data) => {
                    println!("  ✓ Tile z={} x={} y={} ready ({} bytes)", coord.z, coord.x, coord.y, data.len());
                    // Decode JPEG/PNG image
                    if let Ok(img) = image::load_from_memory(data) {
                        let rgba = img.to_rgba8();
                        let (_w, _h) = rgba.dimensions();
                        
                        // Calculate position in atlas based on tile coordinates
                        // Simple equirectangular mapping: x maps to longitude, y maps to latitude
                        let tiles_per_row = 1u32 << coord.z;
                        let tile_width = self.atlas_width / tiles_per_row.max(1);
                        let tile_height = self.atlas_height / tiles_per_row.max(1);
                        
                        let atlas_x = (coord.x * tile_width) % self.atlas_width;
                        let atlas_y = (coord.y * tile_height) % self.atlas_height;
                    
                    // Scale tile to fit atlas slot
                    let scaled = image::imageops::resize(
                        &rgba,
                        tile_width,
                        tile_height,
                        image::imageops::FilterType::Triangle,
                    );
                    
                    // Upload to atlas
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &self.atlas_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: atlas_x,
                                y: atlas_y,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &scaled,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(tile_width * 4),
                            rows_per_image: Some(tile_height),
                        },
                        wgpu::Extent3d {
                            width: tile_width,
                            height: tile_height,
                            depth_or_array_layers: 1,
                        },
                    );
                    
                    self.loaded_tiles.insert(coord, (atlas_x, atlas_y));
                    println!("  📤 Uploaded to atlas at ({}, {})", atlas_x, atlas_y);
                    } else {
                        println!("  ⚠ Failed to decode image for tile z={} x={} y={}", coord.z, coord.x, coord.y);
                    }
                }
                TileStatus::Failed(err) => {
                    println!("  ✗ Tile z={} x={} y={} FAILED: {}", coord.z, coord.x, coord.y, err);
                }
                TileStatus::Pending => {
                    // Still loading
                }
            }
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<crate::tile_fetcher::CacheStats> {
        self.fetcher.cache_stats()
    }
    
    /// Get number of tiles loaded into atlas
    pub fn loaded_tile_count(&self) -> usize {
        self.loaded_tiles.len()
    }
}

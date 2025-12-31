// Phase 8: Tile Texture Array
// GPU texture array for streaming satellite tiles into quadtree chunks
// Constitutional compliance: Doctrine 1 (GPU-first), Doctrine 3 (zero-copy)
#![allow(dead_code)] // Texture array infrastructure wired but not yet rendering

use crate::tile_fetcher::{TileFetcher, TileCoord, TileStatus, GibsConfig};
use std::collections::{HashMap, VecDeque};
use wgpu::util::DeviceExt;

/// Maximum number of texture layers in the array
const MAX_LAYERS: u32 = 128;

/// Tile size in pixels (NASA GIBS tiles are 256x256)
const TILE_SIZE: u32 = 256;

/// Entry in the tile cache
#[derive(Debug, Clone)]
struct TileCacheEntry {
    /// Array layer index (0..MAX_LAYERS)
    layer: u32,
    /// Last frame this tile was accessed
    last_used_frame: u64,
    /// Is the texture data uploaded?
    uploaded: bool,
}

/// Manages a GPU texture array for streaming satellite tiles
pub struct TileTextureArray {
    /// Async tile fetcher
    fetcher: TileFetcher,
    
    /// GPU texture array (256x256 x MAX_LAYERS)
    pub texture_array: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    
    /// Tile indirection: TileCoord → layer index
    tile_to_layer: HashMap<TileCoord, TileCacheEntry>,
    
    /// LRU eviction queue (front = oldest)
    lru_queue: VecDeque<TileCoord>,
    
    /// Free layer indices
    free_layers: Vec<u32>,
    
    /// Current frame number (for LRU tracking)
    current_frame: u64,
    
    /// Pending tile requests
    pending: HashMap<TileCoord, bool>,
}

impl TileTextureArray {
    /// Create new tile texture array manager
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> anyhow::Result<Self> {
        // Create tile fetcher with NASA GIBS config
        let fetcher = TileFetcher::new(GibsConfig::default())?;
        
        // Create 2D texture array (256x256 x MAX_LAYERS)
        let texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tile Texture Array"),
            size: wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: MAX_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Create array view (view_dimension = D2Array)
        let texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Tile Texture Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Tile Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group layout matching globe.wgsl group(1):
        // binding 0: MaterialUniforms (uniform buffer)
        // binding 1: texture_2d_array (satellite_textures) 
        // binding 2: sampler
        // binding 3: texture_2d (fallback single texture)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile Array Bind Group Layout"),
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
        
        // Create material uniform buffer (for shader compatibility)
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
            label: Some("Tile Array Material Buffer"),
            contents: bytemuck::bytes_of(&material_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create fallback 2D texture (binding 3)
        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tile Fallback Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let fallback_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Array Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&fallback_view),
                },
            ],
        });
        
        // Initialize free layer list
        let free_layers: Vec<u32> = (0..MAX_LAYERS).collect();
        
        Ok(Self {
            fetcher,
            texture_array,
            texture_view,
            sampler,
            bind_group,
            bind_group_layout,
            tile_to_layer: HashMap::new(),
            lru_queue: VecDeque::new(),
            free_layers,
            current_frame: 0,
            pending: HashMap::new(),
        })
    }
    
    /// Request a tile for a chunk. Returns layer index if already loaded.
    pub fn request_tile(&mut self, coord: TileCoord) -> Option<i32> {
        // Check if already loaded
        if let Some(entry) = self.tile_to_layer.get_mut(&coord) {
            entry.last_used_frame = self.current_frame;
            if entry.uploaded {
                return Some(entry.layer as i32);
            }
        }
        
        // Check if already pending
        if self.pending.contains_key(&coord) {
            return None;
        }
        
        // Request tile fetch
        self.fetcher.request_tile(coord);
        self.pending.insert(coord, true);
        
        None
    }
    
    /// Get layer index for a tile (if loaded)
    pub fn get_layer(&self, coord: TileCoord) -> Option<i32> {
        self.tile_to_layer.get(&coord)
            .filter(|e| e.uploaded)
            .map(|e| e.layer as i32)
    }
    
    /// Poll for completed tile fetches and upload to GPU
    pub fn update(&mut self, queue: &wgpu::Queue) {
        self.current_frame += 1;
        
        // Poll fetcher for completed tiles
        let results = self.fetcher.poll_results();
        
        // Only log summary, not every tile
        let ready_count = results.iter().filter(|(_, s)| matches!(s, TileStatus::Ready(_))).count();
        if ready_count > 0 {
            println!("🗺️ TileTextureArray: {} tiles ready", ready_count);
        }
        
        for (coord, status) in results {
            self.pending.remove(&coord);
            
            match &status {
                TileStatus::Ready(data) => {
                    // Decode image
                    if let Ok(img) = image::load_from_memory(data) {
                        let rgba = img.to_rgba8();
                    
                        // Get a layer index (allocate or evict)
                        let layer = self.allocate_layer(coord);
                    
                        // Resize tile to TILE_SIZE if needed
                        let (w, h) = rgba.dimensions();
                        let tile_data = if w != TILE_SIZE || h != TILE_SIZE {
                            let resized = image::imageops::resize(
                                &rgba,
                                TILE_SIZE,
                                TILE_SIZE,
                                image::imageops::FilterType::Triangle,
                            );
                            resized.into_raw()
                        } else {
                            rgba.into_raw()
                        };
                    
                        // Upload to texture array layer
                        queue.write_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.texture_array,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: 0,
                                    z: layer,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            &tile_data,
                            wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(TILE_SIZE * 4),
                                rows_per_image: Some(TILE_SIZE),
                            },
                            wgpu::Extent3d {
                                width: TILE_SIZE,
                                height: TILE_SIZE,
                                depth_or_array_layers: 1,
                            },
                        );
                    
                        // Mark as uploaded
                        if let Some(entry) = self.tile_to_layer.get_mut(&coord) {
                            entry.uploaded = true;
                        }
                    }
                }
                TileStatus::Failed(_err) => {
                    // Silently ignore 404s - polar gaps and missing tiles are expected
                }
                TileStatus::Pending => {
                    // Still loading
                }
            }
        }
    }
    
    /// Allocate a layer for a tile (evicting old tiles if needed)
    fn allocate_layer(&mut self, coord: TileCoord) -> u32 {
        // Check if already has a layer
        if let Some(entry) = self.tile_to_layer.get(&coord) {
            return entry.layer;
        }
        
        // Try to get a free layer
        let layer = if let Some(free) = self.free_layers.pop() {
            free
        } else {
            // Evict oldest tile
            self.evict_oldest()
        };
        
        // Create cache entry
        let entry = TileCacheEntry {
            layer,
            last_used_frame: self.current_frame,
            uploaded: false,
        };
        
        self.tile_to_layer.insert(coord, entry);
        self.lru_queue.push_back(coord);
        
        layer
    }
    
    /// Evict the oldest tile and return its layer
    fn evict_oldest(&mut self) -> u32 {
        // Find and remove the oldest tile
        while let Some(old_coord) = self.lru_queue.pop_front() {
            if let Some(entry) = self.tile_to_layer.remove(&old_coord) {
                return entry.layer;
            }
        }
        
        // Fallback: return layer 0 (shouldn't happen)
        0
    }
    
    /// Get statistics about the tile cache
    pub fn stats(&self) -> TileArrayStats {
        TileArrayStats {
            loaded_tiles: self.tile_to_layer.len(),
            pending_tiles: self.pending.len(),
            free_layers: self.free_layers.len(),
            max_layers: MAX_LAYERS as usize,
        }
    }
}

/// Statistics for debugging
#[derive(Debug, Clone)]
pub struct TileArrayStats {
    pub loaded_tiles: usize,
    pub pending_tiles: usize,
    pub free_layers: usize,
    pub max_layers: usize,
}

// Phase 4: NASA GIBS Tile Fetcher
// Asynchronous satellite tile loading with LRU cache
// Constitutional compliance: Doctrine 2 (async, non-blocking), Doctrine 1 (never blocks render)
#![allow(dead_code)] // Infrastructure for Phase 1 NOAA pipeline

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use tokio::sync::mpsc;

/// Tile fetch status
#[derive(Debug, Clone)]
pub enum TileStatus {
    /// Tile is being fetched
    Pending,
    /// Tile is ready
    Ready(Vec<u8>),
    /// Tile fetch failed
    Failed(String),
}

/// WMTS tile coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    /// Zoom level (0-20)
    pub z: u8,
    /// Column (x coordinate)
    pub x: u32,
    /// Row (y coordinate)
    pub y: u32,
}

// Phase 5 scaffolding: TileCoord implementation
#[allow(dead_code)]
impl TileCoord {
    /// Create new tile coordinate
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }
    
    /// Convert lat/lon to tile coordinates at given zoom level
    pub fn from_lat_lon(lat: f64, lon: f64, zoom: u8) -> Self {
        let n = 2.0_f64.powi(zoom as i32);
        let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
        let y = ((1.0 - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
        Self { z: zoom, x, y }
    }
    
    /// Get tile bounds in lat/lon
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let n = 2.0_f64.powi(self.z as i32);
        let lon_min = self.x as f64 / n * 360.0 - 180.0;
        let lon_max = (self.x + 1) as f64 / n * 360.0 - 180.0;
        
        let lat_rad_max = ((1.0 - 2.0 * self.y as f64 / n) * std::f64::consts::PI).sinh().atan();
        let lat_rad_min = ((1.0 - 2.0 * (self.y + 1) as f64 / n) * std::f64::consts::PI).sinh().atan();
        
        let lat_min = lat_rad_min.to_degrees();
        let lat_max = lat_rad_max.to_degrees();
        
        (lat_min, lon_min, lat_max, lon_max)
    }
}

// Phase 5 scaffolding: Tile cache entry for LRU management
#[allow(dead_code)]
/// Tile cache entry
#[derive(Clone)]
pub struct CachedTile {
    /// Raw image data (JPEG or PNG bytes)
    pub data: Vec<u8>,
    /// Last access timestamp (for LRU eviction)
    pub last_access: std::time::Instant,
    /// Tile size in bytes
    pub size_bytes: usize,
}

// Phase 5 scaffolding: LRU tile cache with memory budget
#[allow(dead_code)]
/// LRU tile cache with memory budget
pub struct TileCache {
    /// Cache storage
    tiles: HashMap<TileCoord, CachedTile>,
    /// Maximum cache size in bytes (default: 500 MB)
    max_size_bytes: usize,
    /// Current cache size
    current_size_bytes: usize,
}

// Phase 5 scaffolding: TileCache implementation
#[allow(dead_code)]
impl TileCache {
    /// Create new tile cache with default size (500 MB)
    pub fn new() -> Self {
        Self::with_capacity(500 * 1024 * 1024)
    }
    
    /// Create new tile cache with custom capacity
    pub fn with_capacity(max_size_bytes: usize) -> Self {
        Self {
            tiles: HashMap::new(),
            max_size_bytes,
            current_size_bytes: 0,
        }
    }
    
    /// Get tile from cache (updates LRU)
    pub fn get(&mut self, coord: &TileCoord) -> Option<Vec<u8>> {
        if let Some(entry) = self.tiles.get_mut(coord) {
            entry.last_access = std::time::Instant::now();
            Some(entry.data.clone())
        } else {
            None
        }
    }
    
    /// Insert tile into cache (evicts LRU if needed)
    pub fn insert(&mut self, coord: TileCoord, data: Vec<u8>) {
        let size = data.len();
        
        // Evict LRU entries if cache is full
        while self.current_size_bytes + size > self.max_size_bytes && !self.tiles.is_empty() {
            self.evict_lru();
        }
        
        // Insert new tile
        self.tiles.insert(coord, CachedTile {
            data,
            last_access: std::time::Instant::now(),
            size_bytes: size,
        });
        self.current_size_bytes += size;
    }
    
    /// Evict least recently used tile
    fn evict_lru(&mut self) {
        if let Some((coord, _)) = self.tiles
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, v)| (*k, v.clone()))
        {
            if let Some(entry) = self.tiles.remove(&coord) {
                self.current_size_bytes -= entry.size_bytes;
            }
        }
    }
    
    /// Clear cache
    pub fn clear(&mut self) {
        self.tiles.clear();
        self.current_size_bytes = 0;
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            tile_count: self.tiles.len(),
            size_bytes: self.current_size_bytes,
            max_size_bytes: self.max_size_bytes,
        }
    }
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new()
    }
}

// Phase 5 scaffolding: Cache statistics
#[allow(dead_code)]
/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub tile_count: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
}

// Phase 5 scaffolding: NASA GIBS tile provider configuration
#[allow(dead_code)]
/// NASA GIBS tile provider configuration
#[derive(Debug, Clone)]
pub struct GibsConfig {
    /// Base URL for GIBS WMTS service
    pub base_url: String,
    /// Layer name (e.g., "VIIRS_SNPP_CorrectedReflectance_TrueColor")
    pub layer: String,
    /// Time dimension (ISO 8601 date, e.g., "2024-01-15")
    pub time: String,
    /// Tile matrix set (e.g., "GoogleMapsCompatible_Level9")
    pub tile_matrix_set: String,
    /// Image format (e.g., "jpeg", "png")
    pub format: String,
}

impl Default for GibsConfig {
    fn default() -> Self {
        Self {
            base_url: "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best".to_string(),
            layer: "VIIRS_SNPP_CorrectedReflectance_TrueColor".to_string(),
            time: "2024-01-15".to_string(), // Default to recent date
            tile_matrix_set: "GoogleMapsCompatible_Level9".to_string(),
            format: "jpeg".to_string(),
        }
    }
}

// Phase 5 scaffolding: NASA GIBS tile fetcher
/// NASA GIBS tile fetcher (runs on dedicated thread)
pub struct TileFetcher {
    /// Configuration
    config: GibsConfig,
    /// Tile cache (thread-safe)
    cache: Arc<Mutex<TileCache>>,
    /// Cache directory for persistent storage
    cache_dir: PathBuf,
    /// Channel to send tile requests to background runtime
    request_tx: mpsc::UnboundedSender<TileCoord>,
    /// Channel to receive fetched tiles
    result_rx: Arc<Mutex<mpsc::UnboundedReceiver<(TileCoord, TileStatus)>>>,
}

/// Async tile fetching runtime (runs on background thread)
struct TileFetcherRuntime {
    config: GibsConfig,
    cache: Arc<Mutex<TileCache>>,
    cache_dir: PathBuf,
    request_rx: mpsc::UnboundedReceiver<TileCoord>,
    result_tx: mpsc::UnboundedSender<(TileCoord, TileStatus)>,
    client: reqwest::Client,
    pending: std::collections::HashSet<TileCoord>,
}

impl TileFetcherRuntime {
    async fn run(mut self) {
        while let Some(coord) = self.request_rx.recv().await {
            // Skip if already pending or cached
            if self.pending.contains(&coord) {
                continue;
            }
            if let Ok(cache) = self.cache.lock() {
                if cache.tiles.contains_key(&coord) {
                    continue;
                }
            }
            
            self.pending.insert(coord);
            
            // Check disk cache first
            let cache_path = self.cache_dir.join(format!(
                "{}_{}_{}.{}", coord.z, coord.x, coord.y, self.config.format
            ));
            
            if cache_path.exists() {
                if let Ok(data) = tokio::fs::read(&cache_path).await {
                    let data: Vec<u8> = data;
                    if let Ok(mut cache) = self.cache.lock() {
                        cache.insert(coord, data.clone());
                    }
                    let _ = self.result_tx.send((coord, TileStatus::Ready(data)));
                    self.pending.remove(&coord);
                    continue;
                }
            }
            
            // Fetch from NASA GIBS
            let url = format!(
                "{base}/{layer}/default/{time}/{matrix_set}/{z}/{y}/{x}.{format}",
                base = self.config.base_url,
                layer = self.config.layer,
                time = self.config.time,
                matrix_set = self.config.tile_matrix_set,
                z = coord.z,
                y = coord.y,
                x = coord.x,
                format = self.config.format
            );
            
            match self.client.get(&url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        match response.bytes().await {
                            Ok(bytes) => {
                                let data = bytes.to_vec();
                                
                                // Save to disk cache
                                let _ = tokio::fs::write(&cache_path, &data).await;
                                
                                // Add to memory cache
                                if let Ok(mut cache) = self.cache.lock() {
                                    cache.insert(coord, data.clone());
                                }
                                
                                let _ = self.result_tx.send((coord, TileStatus::Ready(data)));
                            }
                            Err(e) => {
                                let _ = self.result_tx.send((coord, TileStatus::Failed(e.to_string())));
                            }
                        }
                    } else {
                        let _ = self.result_tx.send((coord, TileStatus::Failed(
                            format!("HTTP {}", response.status())
                        )));
                    }
                }
                Err(e) => {
                    let _ = self.result_tx.send((coord, TileStatus::Failed(e.to_string())));
                }
            }
            
            self.pending.remove(&coord);
        }
    }
}

impl TileFetcher {
    /// Create new tile fetcher with background async runtime
    pub fn new(config: GibsConfig) -> Result<Self> {
        let cache_dir = std::env::temp_dir().join("hypertensor_tiles");
        std::fs::create_dir_all(&cache_dir)
            .context("Failed to create tile cache directory")?;
        
        let cache = Arc::new(Mutex::new(TileCache::new()));
        
        // Create channels for async communication
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = mpsc::unbounded_channel();
        
        // Spawn background runtime for async tile fetching
        let runtime_config = config.clone();
        let runtime_cache = Arc::clone(&cache);
        let runtime_cache_dir = cache_dir.clone();
        
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tile fetcher runtime");
            
            let runtime = TileFetcherRuntime {
                config: runtime_config,
                cache: runtime_cache,
                cache_dir: runtime_cache_dir,
                request_rx,
                result_tx,
                client: reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .expect("Failed to create HTTP client"),
                pending: std::collections::HashSet::new(),
            };
            
            rt.block_on(runtime.run());
        });
        
        Ok(Self {
            config,
            cache,
            cache_dir,
            request_tx,
            result_rx: Arc::new(Mutex::new(result_rx)),
        })
    }
    
    /// Request a tile to be fetched asynchronously (non-blocking)
    pub fn request_tile(&self, coord: TileCoord) {
        let _ = self.request_tx.send(coord);
    }
    
    /// Poll for fetched tiles (non-blocking)
    pub fn poll_results(&self) -> Vec<(TileCoord, TileStatus)> {
        let mut results = Vec::new();
        if let Ok(mut rx) = self.result_rx.lock() {
            while let Ok(result) = rx.try_recv() {
                results.push(result);
            }
        }
        results
    }
    
    /// Build GIBS URL for tile (for debugging)
    pub fn build_url(&self, coord: TileCoord) -> String {
        format!(
            "{base}/{layer}/default/{time}/{matrix_set}/{z}/{y}/{x}.{format}",
            base = self.config.base_url,
            layer = self.config.layer,
            time = self.config.time,
            matrix_set = self.config.tile_matrix_set,
            z = coord.z,
            y = coord.y,
            x = coord.x,
            format = self.config.format
        )
    }
    
    /// Get tile from cache (non-blocking)
    pub fn get_tile(&self, coord: TileCoord) -> Option<Vec<u8>> {
        // Try memory cache first
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(data) = cache.get(&coord) {
                return Some(data);
            }
        }
        
        // Check disk cache
        let cache_path = self.cache_path(coord);
        if cache_path.exists() {
            if let Ok(data) = std::fs::read(&cache_path) {
                // Add to memory cache
                if let Ok(mut cache) = self.cache.lock() {
                    cache.insert(coord, data.clone());
                }
                return Some(data);
            }
        }
        
        None
    }
    
    /// Request tiles for visible region based on camera
    /// Note: VIIRS_SNPP_CorrectedReflectance_TrueColor only supports z=0-2
    pub fn request_visible_tiles(&self, lat: f64, lon: f64, zoom: u8) {
        // Clamp zoom to max supported by VIIRS layer (z=0-2)
        let clamped_zoom = zoom.min(2);
        
        // Request center tile and neighbors
        let center = TileCoord::from_lat_lon(lat, lon, clamped_zoom);
        
        // Request 3x3 grid around center for smooth panning
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = (center.x as i32 + dx).max(0) as u32;
                let y = (center.y as i32 + dy).max(0) as u32;
                let max_tile = (1u32 << clamped_zoom) - 1;
                
                if x <= max_tile && y <= max_tile {
                    self.request_tile(TileCoord::new(clamped_zoom, x, y));
                }
            }
        }
    }
    
    /// Get cache file path for tile
    fn cache_path(&self, coord: TileCoord) -> PathBuf {
        self.cache_dir.join(format!(
            "{}_{}_{}.{}", coord.z, coord.x, coord.y, self.config.format
        ))
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.lock().ok().map(|cache| cache.stats())
    }
    
    /// Clear all caches
    pub fn clear_cache(&self) -> Result<()> {
        // Clear memory cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        
        // Clear disk cache
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .context("Failed to clear disk cache")?;
            std::fs::create_dir_all(&self.cache_dir)
                .context("Failed to recreate cache directory")?;
        }
        
        Ok(())
    }
}

// Phase 5 scaffolding: Tile request queue
#[allow(dead_code)]
/// Tile request queue for managing multiple concurrent fetches
pub struct TileRequestQueue {
    /// Pending requests
    pending: Vec<TileCoord>,
    /// Maximum concurrent requests
    max_concurrent: usize,
}

// Phase 5 scaffolding: TileRequestQueue implementation
#[allow(dead_code)]
impl TileRequestQueue {
    /// Create new request queue
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_concurrent,
        }
    }
    
    /// Add tile request
    pub fn request(&mut self, coord: TileCoord) {
        if !self.pending.contains(&coord) {
            self.pending.push(coord);
        }
    }
    
    /// Get next batch of requests
    pub fn next_batch(&mut self) -> Vec<TileCoord> {
        let count = self.max_concurrent.min(self.pending.len());
        self.pending.drain(0..count).collect()
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
    
    /// Get queue length
    pub fn len(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_coord_from_lat_lon() {
        // Test New York City coordinates
        let coord = TileCoord::from_lat_lon(40.7128, -74.0060, 10);
        assert_eq!(coord.z, 10);
        assert!(coord.x > 0);
        assert!(coord.y > 0);
    }
    
    #[test]
    fn test_tile_bounds() {
        let coord = TileCoord::new(5, 10, 12);
        let (lat_min, lon_min, lat_max, lon_max) = coord.bounds();
        
        // Sanity checks
        assert!(lat_min < lat_max);
        assert!(lon_min < lon_max);
        assert!(lat_min >= -90.0 && lat_max <= 90.0);
        assert!(lon_min >= -180.0 && lon_max <= 180.0);
    }
    
    #[test]
    fn test_tile_cache_lru() {
        let mut cache = TileCache::with_capacity(100); // 100 bytes max
        
        // Insert tiles
        cache.insert(TileCoord::new(5, 0, 0), vec![0u8; 40]);
        cache.insert(TileCoord::new(5, 0, 1), vec![1u8; 40]);
        cache.insert(TileCoord::new(5, 0, 2), vec![2u8; 40]);
        
        // Cache should have evicted oldest tile
        assert!(cache.tiles.len() <= 2);
    }
    
    #[test]
    fn test_gibs_url_generation() {
        let config = GibsConfig::default();
        let fetcher = TileFetcher::new(config).unwrap();
        let url = fetcher.build_url(TileCoord::new(5, 10, 12));
        
        assert!(url.contains("gibs.earthdata.nasa.gov"));
        assert!(url.contains("/5/12/10."));
    }
}

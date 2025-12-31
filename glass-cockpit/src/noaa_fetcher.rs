// Phase 1: NOAA Weather Data Fetcher
// Asynchronous S3 fetcher for GFS/HRRR forecast data
// Constitutional compliance: Doctrine 2 (async, non-blocking), Doctrine 7 (tensor format)
#![allow(dead_code)]  // Phase 1 infrastructure - ready for NOAA integration

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use tokio::sync::mpsc;
use chrono::{Utc, DateTime, Duration, Timelike};

/// NOAA forecast model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForecastModel {
    /// Global Forecast System - 28km resolution, global coverage
    Gfs,
    /// High-Resolution Rapid Refresh - 3km resolution, CONUS only
    Hrrr,
}

impl ForecastModel {
    /// Get S3 bucket name (anonymous access)
    pub fn bucket_name(&self) -> &'static str {
        match self {
            ForecastModel::Gfs => "noaa-gfs-bdp-pds",
            ForecastModel::Hrrr => "noaa-hrrr-bdp-pds",
        }
    }
    
    /// Get grid resolution in km
    pub fn resolution_km(&self) -> f32 {
        match self {
            ForecastModel::Gfs => 28.0,
            ForecastModel::Hrrr => 3.0,
        }
    }
    
    /// Get model run frequency in hours
    pub fn run_frequency_hours(&self) -> u32 {
        match self {
            ForecastModel::Gfs => 6,  // 00, 06, 12, 18 UTC
            ForecastModel::Hrrr => 1, // Every hour
        }
    }
    
    /// Check if this model covers the given lat/lon
    pub fn covers(&self, lat: f32, lon: f32) -> bool {
        match self {
            ForecastModel::Gfs => true, // Global
            ForecastModel::Hrrr => {
                // CONUS bounds (approximate)
                (21.0..=53.0).contains(&lat) && (-134.0..=-60.0).contains(&lon)
            }
        }
    }
}

/// Weather variable to fetch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeatherVariable {
    /// U-component of wind (m/s, eastward positive)
    UWind,
    /// V-component of wind (m/s, northward positive)
    VWind,
    /// Vertical velocity (Pa/s, omega - converted to m/s)
    WVelocity,
    /// Temperature (K)
    Temperature,
    /// Surface pressure (Pa)
    Pressure,
    /// Relative humidity (%)
    RelativeHumidity,
    /// Geopotential height (m)
    GeopotentialHeight,
}

impl WeatherVariable {
    /// Get GRIB2 parameter shortName for GFS
    pub fn gfs_param(&self) -> &'static str {
        match self {
            WeatherVariable::UWind => "UGRD",
            WeatherVariable::VWind => "VGRD",
            WeatherVariable::WVelocity => "VVEL",
            WeatherVariable::Temperature => "TMP",
            WeatherVariable::Pressure => "PRES",
            WeatherVariable::RelativeHumidity => "RH",
            WeatherVariable::GeopotentialHeight => "HGT",
        }
    }
    
    /// Get pressure level in hPa (0 = surface)
    pub fn default_level(&self) -> u32 {
        match self {
            WeatherVariable::UWind => 850,
            WeatherVariable::VWind => 850,
            WeatherVariable::WVelocity => 500,
            WeatherVariable::Temperature => 850,
            WeatherVariable::Pressure => 0, // Surface
            WeatherVariable::RelativeHumidity => 850,
            WeatherVariable::GeopotentialHeight => 500,
        }
    }
}

/// Forecast request specification
#[derive(Debug, Clone)]
pub struct ForecastRequest {
    /// Model to use
    pub model: ForecastModel,
    /// Variable to fetch
    pub variable: WeatherVariable,
    /// Pressure level in hPa (0 = surface)
    pub level_hpa: u32,
    /// Forecast hour (0 = analysis, 1+ = forecast)
    pub forecast_hour: u32,
    /// Model run time (None = latest available)
    pub run_time: Option<DateTime<Utc>>,
}

impl ForecastRequest {
    /// Create request for latest available run
    pub fn latest(model: ForecastModel, variable: WeatherVariable) -> Self {
        Self {
            model,
            variable,
            level_hpa: variable.default_level(),
            forecast_hour: 0,
            run_time: None,
        }
    }
    
    /// Build S3 path for this request
    pub fn s3_path(&self, run_time: DateTime<Utc>) -> String {
        match self.model {
            ForecastModel::Gfs => {
                // GFS path: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fHHH
                let date = run_time.format("%Y%m%d");
                let hour = run_time.hour();
                format!(
                    "gfs.{}/{:02}/atmos/gfs.t{:02}z.pgrb2.0p25.f{:03}",
                    date, hour, hour, self.forecast_hour
                )
            }
            ForecastModel::Hrrr => {
                // HRRR path: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2
                let date = run_time.format("%Y%m%d");
                let hour = run_time.hour();
                let domain = if self.level_hpa == 0 { "wrfsfc" } else { "wrfprs" };
                format!(
                    "hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
                    date, hour, domain, self.forecast_hour
                )
            }
        }
    }
    
    /// Get latest model run time for this model
    pub fn latest_run_time(&self) -> DateTime<Utc> {
        let now = Utc::now();
        let freq = self.model.run_frequency_hours();
        
        // Data available ~3-4 hours after model run
        let available = now - Duration::hours(4);
        let run_hour = (available.hour() / freq) * freq;
        
        // Safe: and_hms_opt only fails for invalid h/m/s, run_hour is always valid (0-23)
        available
            .date_naive()
            .and_hms_opt(run_hour, 0, 0)
            .unwrap_or_else(|| {
                // Fallback to midnight if somehow invalid
                available.date_naive().and_hms_opt(0, 0, 0).expect("midnight is always valid")
            })
            .and_utc()
    }
}

/// Fetch status for weather data
#[derive(Debug, Clone)]
pub enum FetchStatus {
    /// Request is pending
    Pending,
    /// Data is ready (raw GRIB2 bytes)
    Ready(Vec<u8>),
    /// Fetch failed
    Failed(String),
}

/// Cache entry for weather data
#[derive(Clone)]
pub struct CachedForecast {
    /// Raw GRIB2 data
    pub data: Vec<u8>,
    /// Fetch timestamp
    pub fetched_at: std::time::Instant,
    /// Data size in bytes
    pub size_bytes: usize,
}

/// LRU cache for weather data
pub struct ForecastCache {
    /// Cache storage (keyed by S3 path)
    entries: HashMap<String, CachedForecast>,
    /// Maximum cache size in bytes (default: 1 GB for weather data)
    max_size_bytes: usize,
    /// Current cache size
    current_size_bytes: usize,
}

impl ForecastCache {
    /// Create new forecast cache (1 GB default)
    pub fn new() -> Self {
        Self::with_capacity(1024 * 1024 * 1024)
    }
    
    /// Create with custom capacity
    pub fn with_capacity(max_size_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_size_bytes,
            current_size_bytes: 0,
        }
    }
    
    /// Get cached data
    pub fn get(&self, path: &str) -> Option<&CachedForecast> {
        self.entries.get(path)
    }
    
    /// Insert data into cache
    pub fn insert(&mut self, path: String, data: Vec<u8>) {
        let size = data.len();
        
        // Evict old entries if needed
        while self.current_size_bytes + size > self.max_size_bytes && !self.entries.is_empty() {
            self.evict_oldest();
        }
        
        self.entries.insert(path, CachedForecast {
            size_bytes: size,
            data,
            fetched_at: std::time::Instant::now(),
        });
        self.current_size_bytes += size;
    }
    
    /// Evict oldest entry
    fn evict_oldest(&mut self) {
        if let Some((path, _)) = self.entries
            .iter()
            .min_by_key(|(_, e)| e.fetched_at)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            if let Some(entry) = self.entries.remove(&path) {
                self.current_size_bytes -= entry.size_bytes;
            }
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.entries.len(), self.current_size_bytes, self.max_size_bytes)
    }
}

impl Default for ForecastCache {
    fn default() -> Self {
        Self::new()
    }
}

/// NOAA weather data fetcher (runs on dedicated async runtime)
pub struct NoaaFetcher {
    /// Forecast cache (thread-safe)
    cache: Arc<Mutex<ForecastCache>>,
    /// Local cache directory for persistent storage
    cache_dir: PathBuf,
    /// Channel to send fetch requests
    request_tx: mpsc::UnboundedSender<ForecastRequest>,
    /// Channel to receive fetched data
    result_rx: Arc<Mutex<mpsc::UnboundedReceiver<(ForecastRequest, FetchStatus)>>>,
    /// Active requests (to avoid duplicates)
    pending: Arc<Mutex<std::collections::HashSet<String>>>,
}

impl NoaaFetcher {
    /// Create new NOAA fetcher with background runtime
    pub fn new() -> Result<Self> {
        let cache_dir = std::env::temp_dir().join("hypertensor_noaa_cache");
        std::fs::create_dir_all(&cache_dir).ok();
        
        let cache = Arc::new(Mutex::new(ForecastCache::new()));
        let pending = Arc::new(Mutex::new(std::collections::HashSet::new()));
        
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = mpsc::unbounded_channel();
        
        // Spawn background runtime
        let runtime = NoaaFetcherRuntime {
            cache: cache.clone(),
            cache_dir: cache_dir.clone(),
            pending: pending.clone(),
            request_rx,
            result_tx,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .context("Failed to create HTTP client")?,
        };
        
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");
            rt.block_on(runtime.run());
        });
        
        Ok(Self {
            cache,
            cache_dir,
            request_tx,
            result_rx: Arc::new(Mutex::new(result_rx)),
            pending,
        })
    }
    
    /// Request forecast data (non-blocking)
    pub fn request(&self, req: ForecastRequest) {
        // Build cache key
        let run_time = req.run_time.unwrap_or_else(|| req.latest_run_time());
        let path = req.s3_path(run_time);
        
        // Check if already pending
        if let Ok(mut pending) = self.pending.lock() {
            if pending.contains(&path) {
                return;
            }
            pending.insert(path.clone());
        }
        
        // Check if already cached
        if let Ok(cache) = self.cache.lock() {
            if cache.get(&path).is_some() {
                return;
            }
        }
        
        // Send request
        let _ = self.request_tx.send(req);
    }
    
    /// Poll for completed fetches (call each frame)
    pub fn poll(&self) -> Vec<(ForecastRequest, FetchStatus)> {
        let mut results = Vec::new();
        
        if let Ok(mut rx) = self.result_rx.lock() {
            while let Ok(result) = rx.try_recv() {
                results.push(result);
            }
        }
        
        results
    }
    
    /// Get cached data if available
    pub fn get_cached(&self, req: &ForecastRequest) -> Option<Vec<u8>> {
        let run_time = req.run_time.unwrap_or_else(|| req.latest_run_time());
        let path = req.s3_path(run_time);
        
        if let Ok(cache) = self.cache.lock() {
            cache.get(&path).map(|e| e.data.clone())
        } else {
            None
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        if let Ok(cache) = self.cache.lock() {
            cache.stats()
        } else {
            (0, 0, 0)
        }
    }
    
    /// Request standard wind field data (U+V at 850hPa)
    pub fn request_wind_field(&self, model: ForecastModel) {
        self.request(ForecastRequest::latest(model, WeatherVariable::UWind));
        self.request(ForecastRequest::latest(model, WeatherVariable::VWind));
    }
    
    /// Request full weather tensor (all standard variables)
    pub fn request_full_tensor(&self, model: ForecastModel) {
        for var in [
            WeatherVariable::UWind,
            WeatherVariable::VWind,
            WeatherVariable::Temperature,
            WeatherVariable::RelativeHumidity,
        ] {
            self.request(ForecastRequest::latest(model, var));
        }
    }
}

/// Background async runtime for NOAA fetching
struct NoaaFetcherRuntime {
    cache: Arc<Mutex<ForecastCache>>,
    cache_dir: PathBuf,
    pending: Arc<Mutex<std::collections::HashSet<String>>>,
    request_rx: mpsc::UnboundedReceiver<ForecastRequest>,
    result_tx: mpsc::UnboundedSender<(ForecastRequest, FetchStatus)>,
    client: reqwest::Client,
}

impl NoaaFetcherRuntime {
    async fn run(mut self) {
        while let Some(req) = self.request_rx.recv().await {
            let run_time = req.run_time.unwrap_or_else(|| req.latest_run_time());
            let path = req.s3_path(run_time);
            
            // Try disk cache first
            let disk_path = self.cache_dir.join(path.replace('/', "_"));
            if disk_path.exists() {
                if let Ok(data) = tokio::fs::read(&disk_path).await {
                    // Store in memory cache
                    if let Ok(mut cache) = self.cache.lock() {
                        cache.insert(path.clone(), data.clone());
                    }
                    
                    // Remove from pending
                    if let Ok(mut pending) = self.pending.lock() {
                        pending.remove(&path);
                    }
                    
                    let _ = self.result_tx.send((req, FetchStatus::Ready(data)));
                    continue;
                }
            }
            
            // Fetch from S3 (HTTPS, anonymous access)
            let url = format!(
                "https://{}.s3.amazonaws.com/{}",
                req.model.bucket_name(),
                path
            );
            
            match self.fetch_with_retry(&url, 3).await {
                Ok(data) => {
                    // Save to disk cache
                    let _ = tokio::fs::write(&disk_path, &data).await;
                    
                    // Store in memory cache
                    if let Ok(mut cache) = self.cache.lock() {
                        cache.insert(path.clone(), data.clone());
                    }
                    
                    // Remove from pending
                    if let Ok(mut pending) = self.pending.lock() {
                        pending.remove(&path);
                    }
                    
                    let _ = self.result_tx.send((req, FetchStatus::Ready(data)));
                }
                Err(e) => {
                    // Remove from pending
                    if let Ok(mut pending) = self.pending.lock() {
                        pending.remove(&path);
                    }
                    
                    let _ = self.result_tx.send((req, FetchStatus::Failed(e.to_string())));
                }
            }
        }
    }
    
    async fn fetch_with_retry(&self, url: &str, max_retries: u32) -> Result<Vec<u8>> {
        let mut last_error = None;
        
        for attempt in 0..max_retries {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(500 * (1 << attempt))).await;
            }
            
            match self.client.get(url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        match response.bytes().await {
                            Ok(bytes) => return Ok(bytes.to_vec()),
                            Err(e) => last_error = Some(e.into()),
                        }
                    } else if response.status().as_u16() == 404 {
                        // Data not yet available - normal for recent model runs
                        return Err(anyhow::anyhow!("Data not yet available (404)"));
                    } else {
                        last_error = Some(anyhow::anyhow!("HTTP {}", response.status()));
                    }
                }
                Err(e) => last_error = Some(e.into()),
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gfs_path_generation() {
        let req = ForecastRequest {
            model: ForecastModel::Gfs,
            variable: WeatherVariable::UWind,
            level_hpa: 850,
            forecast_hour: 6,
            run_time: None,
        };
        
        // Use a fixed time for reproducible test
        let run_time = chrono::NaiveDate::from_ymd_opt(2024, 12, 30)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc();
        
        let path = req.s3_path(run_time);
        assert_eq!(path, "gfs.20241230/12/atmos/gfs.t12z.pgrb2.0p25.f006");
    }
    
    #[test]
    fn test_hrrr_path_generation() {
        let req = ForecastRequest {
            model: ForecastModel::Hrrr,
            variable: WeatherVariable::Temperature,
            level_hpa: 850,
            forecast_hour: 3,
            run_time: None,
        };
        
        let run_time = chrono::NaiveDate::from_ymd_opt(2024, 12, 30)
            .unwrap()
            .and_hms_opt(18, 0, 0)
            .unwrap()
            .and_utc();
        
        let path = req.s3_path(run_time);
        assert_eq!(path, "hrrr.20241230/conus/hrrr.t18z.wrfprsf03.grib2");
    }
    
    #[test]
    fn test_hrrr_coverage() {
        let model = ForecastModel::Hrrr;
        
        // Denver, CO - should be covered
        assert!(model.covers(39.7, -105.0));
        
        // London - should NOT be covered
        assert!(!model.covers(51.5, 0.0));
        
        // Tokyo - should NOT be covered
        assert!(!model.covers(35.7, 139.7));
    }
    
    /// Integration test: Verify NOAA S3 bucket is accessible
    /// Run with: cargo test test_noaa_s3_accessible -- --ignored --nocapture
    #[test]
    #[ignore] // Requires network access
    fn test_noaa_s3_accessible() {
        use std::process::Command;
        
        // Use curl to check S3 accessibility (avoids reqwest blocking feature dependency)
        let gfs_check = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10",
                   "https://noaa-gfs-bdp-pds.s3.amazonaws.com/?list-type=2&max-keys=1"])
            .output();
        
        match gfs_check {
            Ok(output) => {
                let status = String::from_utf8_lossy(&output.stdout);
                println!("GFS S3 bucket HTTP status: {}", status);
                assert!(status.starts_with("200"), "GFS S3 bucket should return 200");
            }
            Err(e) => {
                println!("curl not available or failed: {}", e);
            }
        }
        
        let hrrr_check = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10",
                   "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/?list-type=2&max-keys=1"])
            .output();
        
        match hrrr_check {
            Ok(output) => {
                let status = String::from_utf8_lossy(&output.stdout);
                println!("HRRR S3 bucket HTTP status: {}", status);
                assert!(status.starts_with("200"), "HRRR S3 bucket should return 200");
            }
            Err(e) => {
                println!("curl not available or failed: {}", e);
            }
        }
    }
}

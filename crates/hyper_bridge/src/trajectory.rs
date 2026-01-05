//! Phase 3B-3: Trajectory Protocol for Hypersonic Guidance
//!
//! Streaming protocol for trajectory waypoints via shared memory.
//! 
//! Memory layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ TRAJECTORY HEADER (256 bytes)                                   │
//! │ ├── magic: [u8; 4] = "TRAJ"                                     │
//! │ ├── version: u32 = 1                                            │
//! │ ├── frame_number: u64                                           │
//! │ ├── num_waypoints: u32                                          │
//! │ ├── total_cost: f32                                             │
//! │ ├── path_length: f32                                            │
//! │ ├── start_lat, start_lon, start_alt: f32                        │
//! │ ├── end_lat, end_lon, end_alt: f32                              │
//! │ ├── vehicle_mach: f32                                           │
//! │ ├── timestamp_us: u64                                           │
//! │ └── _padding                                                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ WAYPOINT DATA (num_waypoints × 16 bytes)                        │
//! │ └── [Waypoint { lat: f32, lon: f32, alt: f32, time: f32 }; N]   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Magic number for trajectory protocol: "TRAJ"
pub const TRAJECTORY_MAGIC: [u8; 4] = [b'T', b'R', b'A', b'J'];

/// Trajectory protocol version
pub const TRAJECTORY_VERSION: u32 = 1;

/// Maximum waypoints per trajectory
pub const MAX_WAYPOINTS: usize = 256;

/// Header size (256 bytes, cache-friendly)
pub const TRAJECTORY_HEADER_SIZE: usize = 256;

/// Single trajectory waypoint
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct Waypoint {
    /// Latitude (degrees)
    pub lat: f32,
    /// Longitude (degrees)
    pub lon: f32,
    /// Altitude (meters)
    pub alt: f32,
    /// Time since trajectory start (seconds)
    pub time: f32,
}

impl Waypoint {
    pub fn new(lat: f32, lon: f32, alt: f32, time: f32) -> Self {
        Self { lat, lon, alt, time }
    }
    
    /// Convert to glam Vec3 for rendering
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.lon, self.alt, self.lat)
    }
    
    /// Create from Vec3 (x=lon, y=alt, z=lat)
    pub fn from_vec3(v: Vec3, time: f32) -> Self {
        Self {
            lat: v.z,
            lon: v.x,
            alt: v.y,
            time,
        }
    }
}

/// Trajectory header for IPC streaming
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryHeader {
    /// Magic number "TRAJ"
    pub magic: [u8; 4],
    
    /// Protocol version
    pub version: u32,
    
    /// Frame/update number
    pub frame_number: u64,
    
    /// Number of waypoints in this trajectory
    pub num_waypoints: u32,
    
    /// Optimization flags (bitfield)
    /// - bit 0: converged
    /// - bit 1: has_altitude
    /// - bit 2: is_3d
    pub flags: u32,
    
    // ─────────────────────────────────────────────────────────────────
    // Path Statistics
    // ─────────────────────────────────────────────────────────────────
    
    /// Total integrated cost along path
    pub total_cost: f32,
    
    /// Total path length (meters or normalized)
    pub path_length: f32,
    
    /// Computation time (milliseconds)
    pub computation_time_ms: f32,
    
    /// Number of optimizer iterations
    pub iterations: u32,
    
    // ─────────────────────────────────────────────────────────────────
    // Endpoints
    // ─────────────────────────────────────────────────────────────────
    
    pub start_lat: f32,
    pub start_lon: f32,
    pub start_alt: f32,
    
    pub end_lat: f32,
    pub end_lon: f32,
    pub end_alt: f32,
    
    // ─────────────────────────────────────────────────────────────────
    // Vehicle State
    // ─────────────────────────────────────────────────────────────────
    
    /// Current vehicle Mach number
    pub vehicle_mach: f32,
    
    /// Current vehicle position along trajectory (0-1)
    pub vehicle_progress: f32,
    
    /// Estimated time to destination (seconds)
    pub eta_seconds: f32,
    
    /// Padding for alignment
    _pad0: f32,
    
    // ─────────────────────────────────────────────────────────────────
    // Timestamps
    // ─────────────────────────────────────────────────────────────────
    
    /// Producer timestamp (microseconds since epoch)
    pub producer_timestamp_us: u64,
    
    /// Consumer timestamp (set by Rust when read)
    pub consumer_timestamp_us: u64,
    
    // Remaining padding handled by align(256)
}

// SAFETY: TrajectoryHeader is #[repr(C)] with only primitive types
unsafe impl Pod for TrajectoryHeader {}
unsafe impl Zeroable for TrajectoryHeader {}

impl Default for TrajectoryHeader {
    fn default() -> Self {
        Self {
            magic: TRAJECTORY_MAGIC,
            version: TRAJECTORY_VERSION,
            frame_number: 0,
            num_waypoints: 0,
            flags: 0,
            total_cost: 0.0,
            path_length: 0.0,
            computation_time_ms: 0.0,
            iterations: 0,
            start_lat: 0.0,
            start_lon: 0.0,
            start_alt: 0.0,
            end_lat: 0.0,
            end_lon: 0.0,
            end_alt: 0.0,
            vehicle_mach: 10.0,
            vehicle_progress: 0.0,
            eta_seconds: 0.0,
            _pad0: 0.0,
            producer_timestamp_us: 0,
            consumer_timestamp_us: 0,
        }
    }
}

impl TrajectoryHeader {
    /// Validate the header
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != TRAJECTORY_MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                TRAJECTORY_MAGIC, self.magic
            ));
        }
        
        if self.version != TRAJECTORY_VERSION {
            return Err(format!(
                "Unsupported version: expected {}, got {}",
                TRAJECTORY_VERSION, self.version
            ));
        }
        
        if self.num_waypoints as usize > MAX_WAYPOINTS {
            return Err(format!(
                "Too many waypoints: {} > {}",
                self.num_waypoints, MAX_WAYPOINTS
            ));
        }
        
        Ok(())
    }
    
    /// Check if trajectory converged
    pub fn converged(&self) -> bool {
        self.flags & 0x01 != 0
    }
    
    /// Check if trajectory has altitude data
    pub fn has_altitude(&self) -> bool {
        self.flags & 0x02 != 0
    }
    
    /// Check if this is a 3D trajectory
    pub fn is_3d(&self) -> bool {
        self.flags & 0x04 != 0
    }
    
    /// Set converged flag
    pub fn set_converged(&mut self, val: bool) {
        if val {
            self.flags |= 0x01;
        } else {
            self.flags &= !0x01;
        }
    }
    
    /// Calculate total data size (header + waypoints)
    pub fn total_size(&self) -> usize {
        TRAJECTORY_HEADER_SIZE + (self.num_waypoints as usize) * std::mem::size_of::<Waypoint>()
    }
}

/// Full trajectory data (header + waypoints) for shared memory
#[derive(Debug, Clone)]
pub struct TrajectoryData {
    pub header: TrajectoryHeader,
    pub waypoints: Vec<Waypoint>,
}

impl TrajectoryData {
    /// Create new trajectory data
    pub fn new(waypoints: Vec<Waypoint>) -> Self {
        let mut header = TrajectoryHeader::default();
        header.num_waypoints = waypoints.len() as u32;
        
        if let Some(first) = waypoints.first() {
            header.start_lat = first.lat;
            header.start_lon = first.lon;
            header.start_alt = first.alt;
        }
        
        if let Some(last) = waypoints.last() {
            header.end_lat = last.lat;
            header.end_lon = last.lon;
            header.end_alt = last.alt;
            header.eta_seconds = last.time;
        }
        
        Self { header, waypoints }
    }
    
    /// Convert waypoints to Vec3 for rendering
    pub fn to_vec3_path(&self) -> Vec<Vec3> {
        self.waypoints.iter().map(|w| w.to_vec3()).collect()
    }
    
    /// Serialize to bytes for shared memory
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.header.total_size());
        
        // Header
        bytes.extend_from_slice(bytemuck::bytes_of(&self.header));
        // Pad header to TRAJECTORY_HEADER_SIZE
        bytes.resize(TRAJECTORY_HEADER_SIZE, 0);
        
        // Waypoints
        for wp in &self.waypoints {
            bytes.extend_from_slice(bytemuck::bytes_of(wp));
        }
        
        bytes
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < TRAJECTORY_HEADER_SIZE {
            return Err("Buffer too small for header".to_string());
        }
        
        // Read header
        let header: TrajectoryHeader = *bytemuck::from_bytes(&bytes[..std::mem::size_of::<TrajectoryHeader>()]);
        header.validate()?;
        
        // Read waypoints
        let waypoint_size = std::mem::size_of::<Waypoint>();
        let waypoints_start = TRAJECTORY_HEADER_SIZE;
        let waypoints_end = waypoints_start + (header.num_waypoints as usize) * waypoint_size;
        
        if bytes.len() < waypoints_end {
            return Err("Buffer too small for waypoints".to_string());
        }
        
        let mut waypoints = Vec::with_capacity(header.num_waypoints as usize);
        for i in 0..header.num_waypoints as usize {
            let start = waypoints_start + i * waypoint_size;
            let end = start + waypoint_size;
            let wp: Waypoint = *bytemuck::from_bytes(&bytes[start..end]);
            waypoints.push(wp);
        }
        
        Ok(Self { header, waypoints })
    }
    
    /// Get waypoint at specific time (interpolated)
    pub fn waypoint_at_time(&self, t: f32) -> Option<Waypoint> {
        if self.waypoints.is_empty() {
            return None;
        }
        
        if t <= self.waypoints[0].time {
            return Some(self.waypoints[0]);
        }
        
        for i in 1..self.waypoints.len() {
            if self.waypoints[i].time >= t {
                // Linear interpolation
                let wp0 = &self.waypoints[i - 1];
                let wp1 = &self.waypoints[i];
                let alpha = (t - wp0.time) / (wp1.time - wp0.time);
                
                return Some(Waypoint {
                    lat: wp0.lat + alpha * (wp1.lat - wp0.lat),
                    lon: wp0.lon + alpha * (wp1.lon - wp0.lon),
                    alt: wp0.alt + alpha * (wp1.alt - wp0.alt),
                    time: t,
                });
            }
        }
        
        Some(*self.waypoints.last().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_waypoint_size() {
        assert_eq!(std::mem::size_of::<Waypoint>(), 16);
    }
    
    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<TrajectoryHeader>(), TRAJECTORY_HEADER_SIZE);
    }
    
    #[test]
    fn test_trajectory_roundtrip() {
        let waypoints = vec![
            Waypoint::new(30.0, -100.0, 25000.0, 0.0),
            Waypoint::new(31.0, -99.0, 27000.0, 60.0),
            Waypoint::new(32.0, -98.0, 29000.0, 120.0),
        ];
        
        let traj = TrajectoryData::new(waypoints);
        let bytes = traj.to_bytes();
        let restored = TrajectoryData::from_bytes(&bytes).unwrap();
        
        assert_eq!(traj.header.num_waypoints, restored.header.num_waypoints);
        assert_eq!(traj.waypoints.len(), restored.waypoints.len());
        assert_eq!(traj.waypoints[0].lat, restored.waypoints[0].lat);
    }
    
    #[test]
    fn test_interpolation() {
        let waypoints = vec![
            Waypoint::new(0.0, 0.0, 0.0, 0.0),
            Waypoint::new(10.0, 10.0, 1000.0, 10.0),
        ];
        
        let traj = TrajectoryData::new(waypoints);
        let mid = traj.waypoint_at_time(5.0).unwrap();
        
        assert!((mid.lat - 5.0).abs() < 0.01);
        assert!((mid.lon - 5.0).abs() < 0.01);
        assert!((mid.alt - 500.0).abs() < 0.1);
    }
}

/*!
 * Scenario Injection Buffer - Phase 8: Appendix E
 * 
 * Asynchronous injection buffer for user-triggered scenario modifications.
 * The simulation polls this buffer at its own cadence - never waiting.
 * 
 * Memory Layout (per Appendix E.2.1):
 * ```
 * OFFSET    SIZE    TYPE        DESCRIPTION
 * 0x0000    4       u32         Magic number (0x494E4A42 = "INJB")
 * 0x0004    4       u32         Version (1)
 * 0x0008    4       u32         Injection count (monotonic)
 * 0x000C    4       u32         Pending flag (0 = empty, 1 = pending)
 * 0x0010    4       u32         Injection type enum
 * 0x0014    12      f32[3]      Target position (lat, lon, alt)
 * 0x0020    4       f32         Magnitude
 * 0x0024    4       f32         Radius (km)
 * 0x0028    4       f32         Duration (frames)
 * 0x002C    4       u32         Acknowledged flag (sim sets to 1)
 * ```
 * 
 * Constitutional Compliance:
 * - Article II: Simulation never waits for display
 * - Doctrine 2: Uses shared memory, not network
 */

/// Magic number identifying injection buffer
pub const INJECTION_MAGIC: u32 = 0x494E4A42; // "INJB"

/// Buffer version
pub const INJECTION_VERSION: u32 = 1;

/// Maximum injection magnitude (prevents simulation instability)
pub const MAX_INJECTION_MAGNITUDE: f32 = 100.0;

/// Maximum injection radius in km
pub const MAX_INJECTION_RADIUS: f32 = 1000.0;

/// Injection type enumeration
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InjectionType {
    None = 0,
    HeatPulse = 1,        // Temperature anomaly
    PressureDrop = 2,     // Barometric collapse
    MoistureInject = 3,   // Humidity spike
    VorticityForce = 4,   // Rotational impulse
    CustomTensor = 5,     // Raw tensor override (advanced)
}

impl From<u32> for InjectionType {
    fn from(v: u32) -> Self {
        match v {
            1 => InjectionType::HeatPulse,
            2 => InjectionType::PressureDrop,
            3 => InjectionType::MoistureInject,
            4 => InjectionType::VorticityForce,
            5 => InjectionType::CustomTensor,
            _ => InjectionType::None,
        }
    }
}

/// Injection request from UI
#[derive(Clone, Debug)]
pub struct InjectionRequest {
    pub injection_type: InjectionType,
    pub target_lat: f32,
    pub target_lon: f32,
    pub target_alt: f32,
    pub magnitude: f32,
    pub radius_km: f32,
    pub duration_frames: u32,
}

/// Injection errors
#[derive(Clone, Debug)]
pub enum InjectionError {
    MagnitudeExceeded,
    RadiusExceeded,
    BufferBusy,
    ShmNotAvailable,
    InvalidMagic,
}

/// In-memory injection buffer for local testing
/// Production would use shared memory at /dev/shm/sovereign_injection
pub struct InjectionBuffer {
    // Simulated shared memory (in real impl, this would be mmap'd)
    magic: u32,
    version: u32,
    injection_count: u32,
    pending_flag: u32,
    injection_type: u32,
    target_lat: f32,
    target_lon: f32,
    target_alt: f32,
    magnitude: f32,
    radius_km: f32,
    duration_frames: f32,
    acknowledged: u32,
}

impl InjectionBuffer {
    /// Create new injection buffer
    pub fn new() -> Self {
        Self {
            magic: INJECTION_MAGIC,
            version: INJECTION_VERSION,
            injection_count: 0,
            pending_flag: 0,
            injection_type: 0,
            target_lat: 0.0,
            target_lon: 0.0,
            target_alt: 0.0,
            magnitude: 0.0,
            radius_km: 0.0,
            duration_frames: 0.0,
            acknowledged: 0,
        }
    }
    
    /// Submit an injection request
    /// Returns error if validation fails or buffer is busy
    pub fn submit(&mut self, request: InjectionRequest) -> Result<(), InjectionError> {
        // 1. Validate magic (in real impl, check shm header)
        if self.magic != INJECTION_MAGIC {
            return Err(InjectionError::InvalidMagic);
        }
        
        // 2. Validate parameters
        if request.magnitude.abs() > MAX_INJECTION_MAGNITUDE {
            return Err(InjectionError::MagnitudeExceeded);
        }
        if request.radius_km > MAX_INJECTION_RADIUS {
            return Err(InjectionError::RadiusExceeded);
        }
        
        // 3. Check if previous injection still pending
        if self.pending_flag == 1 {
            return Err(InjectionError::BufferBusy);
        }
        
        // 4. Write parameters (ORDER MATTERS - pending flag last)
        self.injection_type = request.injection_type as u32;
        self.target_lat = request.target_lat;
        self.target_lon = request.target_lon;
        self.target_alt = request.target_alt;
        self.magnitude = request.magnitude;
        self.radius_km = request.radius_km;
        self.duration_frames = request.duration_frames as f32;
        self.acknowledged = 0;
        
        // Memory barrier would go here in real impl
        
        // 5. Set pending flag (signals to simulation)
        self.pending_flag = 1;
        self.injection_count += 1;
        
        Ok(())
    }
    
    /// Check if injection was acknowledged by simulation
    pub fn is_acknowledged(&self) -> bool {
        self.acknowledged == 1
    }
    
    /// Check if buffer is busy (injection pending)
    pub fn is_pending(&self) -> bool {
        self.pending_flag == 1
    }
    
    /// Get injection count
    pub fn injection_count(&self) -> u32 {
        self.injection_count
    }
    
    /// Simulate simulation consuming the injection (for testing)
    #[allow(dead_code)]
    pub fn consume(&mut self) {
        if self.pending_flag == 1 {
            self.acknowledged = 1;
            self.pending_flag = 0;
        }
    }
    
    /// Read pending injection (simulation side)
    pub fn read_pending(&self) -> Option<InjectionRequest> {
        if self.pending_flag != 1 {
            return None;
        }
        
        Some(InjectionRequest {
            injection_type: InjectionType::from(self.injection_type),
            target_lat: self.target_lat,
            target_lon: self.target_lon,
            target_alt: self.target_alt,
            magnitude: self.magnitude,
            radius_km: self.radius_km,
            duration_frames: self.duration_frames as u32,
        })
    }
}

impl Default for InjectionBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_injection_submit() {
        let mut buffer = InjectionBuffer::new();
        
        let request = InjectionRequest {
            injection_type: InjectionType::HeatPulse,
            target_lat: 45.0,
            target_lon: -122.0,
            target_alt: 1000.0,
            magnitude: 10.0,
            radius_km: 50.0,
            duration_frames: 60,
        };
        
        assert!(buffer.submit(request).is_ok());
        assert!(buffer.is_pending());
        assert_eq!(buffer.injection_count(), 1);
    }
    
    #[test]
    fn test_injection_validation() {
        let mut buffer = InjectionBuffer::new();
        
        // Magnitude too high
        let bad_request = InjectionRequest {
            injection_type: InjectionType::HeatPulse,
            target_lat: 0.0,
            target_lon: 0.0,
            target_alt: 0.0,
            magnitude: 200.0, // Exceeds MAX
            radius_km: 50.0,
            duration_frames: 60,
        };
        
        assert!(matches!(
            buffer.submit(bad_request),
            Err(InjectionError::MagnitudeExceeded)
        ));
    }
    
    #[test]
    fn test_injection_busy() {
        let mut buffer = InjectionBuffer::new();
        
        let request = InjectionRequest {
            injection_type: InjectionType::PressureDrop,
            target_lat: 0.0,
            target_lon: 0.0,
            target_alt: 0.0,
            magnitude: 5.0,
            radius_km: 100.0,
            duration_frames: 30,
        };
        
        buffer.submit(request.clone()).unwrap();
        
        // Second submission should fail (buffer busy)
        assert!(matches!(
            buffer.submit(request),
            Err(InjectionError::BufferBusy)
        ));
    }
    
    #[test]
    fn test_injection_consume() {
        let mut buffer = InjectionBuffer::new();
        
        let request = InjectionRequest {
            injection_type: InjectionType::VorticityForce,
            target_lat: 30.0,
            target_lon: 60.0,
            target_alt: 500.0,
            magnitude: 8.0,
            radius_km: 200.0,
            duration_frames: 120,
        };
        
        buffer.submit(request).unwrap();
        assert!(buffer.is_pending());
        
        // Simulate consumption
        buffer.consume();
        assert!(!buffer.is_pending());
        assert!(buffer.is_acknowledged());
    }
}

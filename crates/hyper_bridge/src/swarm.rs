//! Swarm Protocol
//! ===============
//! 
//! Multi-agent IPC protocol for coordinated hypersonic flight.
//! 
//! Memory Layout: [SwarmHeader] + [EntityState * N]
//! 
//! This enables the Glass Cockpit to render multiple autonomous
//! agents (Blue Force) executing coordinated maneuvers.

use bytemuck::{Pod, Zeroable};

/// Magic number for protocol validation
pub const SWARM_MAGIC: u32 = 0x5357524D; // "SWRM"

/// Protocol version
pub const SWARM_VERSION: u32 = 1;

/// Maximum entities per swarm
pub const MAX_SWARM_SIZE: usize = 64;

// =============================================================================
// SWARM HEADER
// =============================================================================

/// Header for swarm state transmission.
/// 
/// Fixed size: 64 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SwarmHeader {
    /// Magic number for validation (0x5357524D = "SWRM")
    pub magic: u32,
    
    /// Protocol version
    pub version: u32,
    
    /// Number of entities in this update
    pub entity_count: u32,
    
    /// Padding for alignment (u64 needs 8-byte alignment)
    pub _pad0: u32,
    
    /// Simulation timestamp (microseconds since start)
    pub timestamp: u64,
    
    /// Frame sequence number (for detecting drops)
    pub sequence: u64,
    
    /// Swarm formation type (0=none, 1=line, 2=wedge, 3=echelon, 4=custom)
    pub formation: u32,
    
    /// Mission phase (0=idle, 1=ingress, 2=attack, 3=egress)
    pub mission_phase: u32,
    
    /// Alert level (0=green, 1=yellow, 2=red)
    pub alert_level: u32,
    
    /// Reserved for future use
    pub _padding: [u32; 5],
}

impl SwarmHeader {
    /// Create a new swarm header.
    pub fn new(entity_count: u32, timestamp: u64) -> Self {
        Self {
            magic: SWARM_MAGIC,
            version: SWARM_VERSION,
            entity_count,
            _pad0: 0,
            timestamp,
            sequence: 0,
            formation: 0,
            mission_phase: 0,
            alert_level: 0,
            _padding: [0; 5],
        }
    }
    
    /// Validate the header.
    pub fn is_valid(&self) -> bool {
        self.magic == SWARM_MAGIC && self.version == SWARM_VERSION
    }
    
    /// Size of the complete message (header + entities).
    pub fn message_size(&self) -> usize {
        std::mem::size_of::<SwarmHeader>() 
            + (self.entity_count as usize) * std::mem::size_of::<EntityState>()
    }
    
    /// Convert to bytes for IPC.
    pub fn to_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(self).to_vec()
    }
    
    /// Parse from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<SwarmHeader>() {
            return None;
        }
        let header: Self = *bytemuck::from_bytes(&bytes[..std::mem::size_of::<Self>()]);
        if header.is_valid() {
            Some(header)
        } else {
            None
        }
    }
}

// =============================================================================
// ENTITY STATE
// =============================================================================

/// Entity type classification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    /// Friendly hypersonic vehicle
    BlueForce = 0,
    /// Hostile target
    RedForce = 1,
    /// Neutral/Unknown
    Neutral = 2,
    /// Objective/Waypoint marker
    Objective = 3,
}

/// State of a single entity in the swarm.
/// 
/// Fixed size: 64 bytes (cache-line aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EntityState {
    /// Unique entity identifier
    pub id: u32,
    
    /// Entity type (0=blue, 1=red, 2=neutral, 3=objective)
    pub entity_type: u8,
    
    /// Status flags (bit 0: alive, bit 1: selected, bit 2: leader)
    pub flags: u8,
    
    /// Formation slot (0 = leader, 1-N = wingmen)
    pub formation_slot: u8,
    
    /// Team/squadron ID
    pub team: u8,
    
    /// Position in world coordinates (meters)
    pub position: [f32; 3],
    
    /// Velocity (m/s)
    pub velocity: [f32; 3],
    
    /// Orientation as quaternion [x, y, z, w]
    pub orientation: [f32; 4],
    
    /// Heat load (0.0 = cool, 1.0 = TPS limit)
    pub heat_load: f32,
    
    /// Current Mach number
    pub mach: f32,
    
    /// Altitude (meters)
    pub altitude: f32,
    
    /// Distance from trajectory tube center (meters)
    pub tube_distance: f32,
}

impl EntityState {
    /// Create a new entity state.
    pub fn new(id: u32, entity_type: EntityType) -> Self {
        Self {
            id,
            entity_type: entity_type as u8,
            flags: 0x01, // Alive
            formation_slot: 0,
            team: 0,
            position: [0.0, 0.0, 30000.0], // 30km altitude
            velocity: [3000.0, 0.0, 0.0],   // Mach 10
            orientation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            heat_load: 0.0,
            mach: 10.0,
            altitude: 30000.0,
            tube_distance: 0.0,
        }
    }
    
    /// Check if entity is alive.
    pub fn is_alive(&self) -> bool {
        (self.flags & 0x01) != 0
    }
    
    /// Check if entity is selected (for UI).
    pub fn is_selected(&self) -> bool {
        (self.flags & 0x02) != 0
    }
    
    /// Check if entity is the formation leader.
    pub fn is_leader(&self) -> bool {
        (self.flags & 0x04) != 0
    }
    
    /// Get entity type.
    pub fn get_type(&self) -> EntityType {
        match self.entity_type {
            0 => EntityType::BlueForce,
            1 => EntityType::RedForce,
            2 => EntityType::Neutral,
            3 => EntityType::Objective,
            _ => EntityType::Neutral,
        }
    }
    
    /// Calculate speed from velocity.
    pub fn speed(&self) -> f32 {
        let [vx, vy, vz] = self.velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }
    
    /// Convert to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(self).to_vec()
    }
}

// Compile-time size assertions (Constitutional Article VIII)
// Both SwarmHeader and EntityState are 64 bytes = cache line aligned
const _: () = {
    assert!(std::mem::size_of::<SwarmHeader>() == 64);
    assert!(std::mem::size_of::<EntityState>() == 64);
    assert!(std::mem::size_of::<SwarmHeader>().is_power_of_two());
    assert!(std::mem::size_of::<EntityState>().is_power_of_two());
    assert!(std::mem::size_of::<CommandMessage>().is_power_of_two());
};

// =============================================================================
// SWARM DATA
// =============================================================================

/// Complete swarm state for IPC transmission.
pub struct SwarmData {
    pub header: SwarmHeader,
    pub entities: Vec<EntityState>,
}

impl SwarmData {
    /// Create a new swarm with the given entities.
    pub fn new(entities: Vec<EntityState>, timestamp: u64) -> Self {
        let header = SwarmHeader::new(entities.len() as u32, timestamp);
        Self { header, entities }
    }
    
    /// Create an empty swarm.
    pub fn empty() -> Self {
        Self {
            header: SwarmHeader::new(0, 0),
            entities: Vec::new(),
        }
    }
    
    /// Serialize to bytes for IPC.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.header.to_bytes();
        for entity in &self.entities {
            bytes.extend(entity.to_bytes());
        }
        bytes
    }
    
    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let header = SwarmHeader::from_bytes(bytes)?;
        
        let entity_size = std::mem::size_of::<EntityState>();
        let header_size = std::mem::size_of::<SwarmHeader>();
        let expected_size = header_size + (header.entity_count as usize) * entity_size;
        
        if bytes.len() < expected_size {
            return None;
        }
        
        let mut entities = Vec::with_capacity(header.entity_count as usize);
        for i in 0..header.entity_count as usize {
            let start = header_size + i * entity_size;
            let end = start + entity_size;
            let entity: EntityState = *bytemuck::from_bytes(&bytes[start..end]);
            entities.push(entity);
        }
        
        Some(Self { header, entities })
    }
    
    /// Get entity by ID.
    pub fn get_entity(&self, id: u32) -> Option<&EntityState> {
        self.entities.iter().find(|e| e.id == id)
    }
    
    /// Get mutable entity by ID.
    pub fn get_entity_mut(&mut self, id: u32) -> Option<&mut EntityState> {
        self.entities.iter_mut().find(|e| e.id == id)
    }
    
    /// Count alive entities.
    pub fn alive_count(&self) -> usize {
        self.entities.iter().filter(|e| e.is_alive()).count()
    }
    
    /// Get the formation leader.
    pub fn get_leader(&self) -> Option<&EntityState> {
        self.entities.iter().find(|e| e.is_leader())
    }
}

// =============================================================================
// SWARM FORMATIONS
// =============================================================================

/// Formation types for coordinated flight.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formation {
    /// No formation, free flight
    None = 0,
    /// Line abreast
    Line = 1,
    /// V formation (wedge)
    Wedge = 2,
    /// Echelon (diagonal line)
    Echelon = 3,
    /// Custom waypoints
    Custom = 4,
}

impl Formation {
    /// Calculate offset for a formation slot.
    /// 
    /// Returns (lateral, longitudinal) offset in meters.
    pub fn slot_offset(&self, slot: u8, spacing: f32) -> (f32, f32) {
        match self {
            Formation::None => (0.0, 0.0),
            
            Formation::Line => {
                // Line abreast: all entities side by side
                let lateral = (slot as f32 - 2.0) * spacing;
                (lateral, 0.0)
            }
            
            Formation::Wedge => {
                // V formation: leader at front
                if slot == 0 {
                    (0.0, 0.0)
                } else {
                    let side = if slot % 2 == 1 { 1.0 } else { -1.0 };
                    let row = ((slot + 1) / 2) as f32;
                    let lateral = side * row * spacing;
                    let longitudinal = -row * spacing;
                    (lateral, longitudinal)
                }
            }
            
            Formation::Echelon => {
                // Diagonal line trailing right
                let offset = slot as f32;
                (offset * spacing * 0.5, -offset * spacing)
            }
            
            Formation::Custom => (0.0, 0.0),
        }
    }
}

// =============================================================================
// COMMANDS
// =============================================================================

/// Commands that can be sent to the swarm.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmCommand {
    /// Hold current position/trajectory
    Hold = 0,
    /// Proceed to next waypoint
    Proceed = 1,
    /// Execute attack run
    Attack = 2,
    /// Break formation and evade
    Evade = 3,
    /// Return to base
    RTB = 4,
    /// Change formation
    Formation = 5,
    /// Intercept target
    Intercept = 6,
}

/// Command message structure.
/// 
/// Fixed size: 64 bytes (cache-line aligned, power-of-2)
/// Layout: 4 + 4 + 24 + 8 + 24 = 64 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CommandMessage {
    /// Command type
    pub command: u32,
    
    /// Target entity ID (0 = all)
    pub target_id: u32,
    
    /// Command parameters
    pub params: [f32; 6],
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Padding to 64 bytes (power-of-2)
    pub _padding: [u8; 24],
}

impl CommandMessage {
    /// Create a new command.
    pub fn new(command: SwarmCommand, target_id: u32) -> Self {
        Self {
            command: command as u32,
            target_id,
            params: [0.0; 6],
            timestamp: 0,
            _padding: [0; 24],
        }
    }
    
    /// Create an intercept command.
    pub fn intercept(target_id: u32, heading: f32, mach: f32) -> Self {
        let mut cmd = Self::new(SwarmCommand::Intercept, target_id);
        cmd.params[0] = heading;
        cmd.params[1] = mach;
        cmd
    }
    
    /// Create a formation change command.
    pub fn formation(formation: Formation, spacing: f32) -> Self {
        let mut cmd = Self::new(SwarmCommand::Formation, 0);
        cmd.params[0] = formation as u32 as f32;
        cmd.params[1] = spacing;
        cmd
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swarm_header_size() {
        assert_eq!(std::mem::size_of::<SwarmHeader>(), 64);
    }
    
    #[test]
    fn test_entity_state_size() {
        assert_eq!(std::mem::size_of::<EntityState>(), 64);
    }
    
    #[test]
    fn test_swarm_serialization() {
        let mut entities = vec![
            EntityState::new(1, EntityType::BlueForce),
            EntityState::new(2, EntityType::BlueForce),
            EntityState::new(3, EntityType::BlueForce),
        ];
        
        // Set leader
        entities[0].flags |= 0x04;
        entities[0].formation_slot = 0;
        entities[1].formation_slot = 1;
        entities[2].formation_slot = 2;
        
        let swarm = SwarmData::new(entities, 1000000);
        
        // Serialize
        let bytes = swarm.to_bytes();
        assert_eq!(bytes.len(), 64 + 3 * 64); // Header + 3 entities
        
        // Deserialize
        let recovered = SwarmData::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.header.entity_count, 3);
        assert_eq!(recovered.entities.len(), 3);
        assert!(recovered.get_leader().is_some());
        assert_eq!(recovered.get_leader().unwrap().id, 1);
    }
    
    #[test]
    fn test_wedge_formation() {
        let formation = Formation::Wedge;
        let spacing = 100.0;
        
        // Leader at origin
        assert_eq!(formation.slot_offset(0, spacing), (0.0, 0.0));
        
        // First wingman right-back
        let (lat, lon) = formation.slot_offset(1, spacing);
        assert!(lat > 0.0);
        assert!(lon < 0.0);
        
        // Second wingman left-back
        let (lat, lon) = formation.slot_offset(2, spacing);
        assert!(lat < 0.0);
        assert!(lon < 0.0);
    }
}

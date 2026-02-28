//! Swarm Renderer
//! ===============
//! 
//! Renders multiple autonomous agents on the Glass Cockpit.
//! 
//! Features:
//! - Blue Force: Friendly hypersonic vehicles
//! - Formation visualization (wedge, echelon, line)
//! - Health/heat indicators
//! - Trail effects per entity
//! - Leader highlighting

use glam::{Vec3, Vec4, Quat};
use ontic_bridge::swarm::{EntityState, EntityType, SwarmData, Formation};

// =============================================================================
// SWARM RENDERER CONFIG
// =============================================================================

/// Configuration for swarm rendering.
#[derive(Debug, Clone)]
pub struct SwarmRenderConfig {
    /// Size of each entity mesh (meters)
    pub entity_scale: f32,
    
    /// Trail length (number of positions)
    pub trail_length: usize,
    
    /// Trail fade alpha
    pub trail_alpha: f32,
    
    /// Formation line color
    pub formation_color: [f32; 4],
    
    /// Blue force color
    pub blue_color: [f32; 4],
    
    /// Red force color
    pub red_color: [f32; 4],
    
    /// Selected entity highlight color
    pub selected_color: [f32; 4],
    
    /// Leader marker color
    pub leader_color: [f32; 4],
    
    /// Heat warning threshold (0-1)
    pub heat_warning: f32,
    
    /// Heat critical threshold (0-1)
    pub heat_critical: f32,
}

impl Default for SwarmRenderConfig {
    fn default() -> Self {
        Self {
            entity_scale: 50.0,  // 50m wingspan equivalent
            trail_length: 50,
            trail_alpha: 0.3,
            formation_color: [0.0, 1.0, 0.5, 0.5],
            blue_color: [0.2, 0.5, 1.0, 1.0],
            red_color: [1.0, 0.2, 0.2, 1.0],
            selected_color: [1.0, 1.0, 0.0, 1.0],
            leader_color: [0.0, 1.0, 0.0, 1.0],
            heat_warning: 0.6,
            heat_critical: 0.85,
        }
    }
}

// =============================================================================
// ENTITY TRAIL
// =============================================================================

/// Trail history for a single entity.
pub struct EntityTrail {
    /// Historical positions
    positions: Vec<Vec3>,
    
    /// Maximum length
    max_length: usize,
}

impl EntityTrail {
    pub fn new(max_length: usize) -> Self {
        Self {
            positions: Vec::with_capacity(max_length),
            max_length,
        }
    }
    
    /// Add a new position to the trail.
    pub fn push(&mut self, pos: Vec3) {
        self.positions.push(pos);
        if self.positions.len() > self.max_length {
            self.positions.remove(0);
        }
    }
    
    /// Get trail positions with alpha values.
    pub fn get_trail_with_alpha(&self) -> Vec<(Vec3, f32)> {
        let n = self.positions.len();
        self.positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                let alpha = (i as f32 + 1.0) / n as f32;
                (pos, alpha)
            })
            .collect()
    }
    
    /// Clear the trail.
    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

// =============================================================================
// SWARM VISUAL STATE
// =============================================================================

/// Visual state for a single entity.
pub struct EntityVisual {
    /// Entity ID
    pub id: u32,
    
    /// Current transform
    pub position: Vec3,
    pub orientation: Quat,
    pub scale: Vec3,
    
    /// Color (may change based on heat/selection)
    pub color: Vec4,
    
    /// Trail history
    pub trail: EntityTrail,
    
    /// Heat indicator (0-1)
    pub heat: f32,
    
    /// Is selected
    pub selected: bool,
    
    /// Is leader
    pub is_leader: bool,
}

impl EntityVisual {
    pub fn new(id: u32, scale: f32, trail_length: usize) -> Self {
        Self {
            id,
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            scale: Vec3::splat(scale),
            color: Vec4::new(0.2, 0.5, 1.0, 1.0),
            trail: EntityTrail::new(trail_length),
            heat: 0.0,
            selected: false,
            is_leader: false,
        }
    }
    
    /// Update from entity state.
    pub fn update(&mut self, state: &EntityState, config: &SwarmRenderConfig) {
        // Update position
        let new_pos = Vec3::from_array(state.position);
        self.trail.push(new_pos);
        self.position = new_pos;
        
        // Update orientation
        let q = state.orientation;
        self.orientation = Quat::from_xyzw(q[0], q[1], q[2], q[3]);
        
        // Update heat
        self.heat = state.heat_load;
        
        // Update flags
        self.selected = state.is_selected();
        self.is_leader = state.is_leader();
        
        // Calculate color based on state
        self.color = self.calculate_color(state, config);
    }
    
    fn calculate_color(&self, state: &EntityState, config: &SwarmRenderConfig) -> Vec4 {
        // Base color by type
        let base = match state.get_type() {
            EntityType::BlueForce => Vec4::from_array(config.blue_color),
            EntityType::RedForce => Vec4::from_array(config.red_color),
            EntityType::Neutral => Vec4::new(0.7, 0.7, 0.7, 1.0),
            EntityType::Objective => Vec4::new(1.0, 0.8, 0.0, 1.0),
        };
        
        // Modify by heat
        let heat = state.heat_load;
        let heat_color = if heat > config.heat_critical {
            Vec4::new(1.0, 0.0, 0.0, 1.0) // Critical: red
        } else if heat > config.heat_warning {
            Vec4::new(1.0, 0.5, 0.0, 1.0) // Warning: orange
        } else {
            base
        };
        
        // Highlight if selected
        if state.is_selected() {
            Vec4::from_array(config.selected_color)
        } else if state.is_leader() {
            heat_color.lerp(Vec4::from_array(config.leader_color), 0.3)
        } else {
            heat_color
        }
    }
}

// =============================================================================
// SWARM RENDERER
// =============================================================================

/// Renderer for the entire swarm.
pub struct SwarmRenderer {
    /// Configuration
    pub config: SwarmRenderConfig,
    
    /// Visual states for each entity
    entities: std::collections::HashMap<u32, EntityVisual>,
    
    /// Current formation
    pub formation: Formation,
    
    /// Formation spacing
    pub formation_spacing: f32,
}

impl SwarmRenderer {
    pub fn new(config: SwarmRenderConfig) -> Self {
        Self {
            config,
            entities: std::collections::HashMap::new(),
            formation: Formation::None,
            formation_spacing: 100.0,
        }
    }
    
    /// Update from swarm data.
    pub fn update(&mut self, swarm: &SwarmData) {
        // Update formation
        self.formation = match swarm.header.formation {
            0 => Formation::None,
            1 => Formation::Line,
            2 => Formation::Wedge,
            3 => Formation::Echelon,
            _ => Formation::Custom,
        };
        
        // Mark all entities as potentially stale
        let mut seen_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        
        // Update entities
        for state in &swarm.entities {
            seen_ids.insert(state.id);
            
            let visual = self.entities.entry(state.id).or_insert_with(|| {
                EntityVisual::new(
                    state.id,
                    self.config.entity_scale,
                    self.config.trail_length,
                )
            });
            
            visual.update(state, &self.config);
        }
        
        // Remove entities that are no longer in the swarm
        self.entities.retain(|id, _| seen_ids.contains(id));
    }
    
    /// Get all entity visuals.
    pub fn get_entities(&self) -> impl Iterator<Item = &EntityVisual> {
        self.entities.values()
    }
    
    /// Get entity by ID.
    pub fn get_entity(&self, id: u32) -> Option<&EntityVisual> {
        self.entities.get(&id)
    }
    
    /// Get the formation leader.
    pub fn get_leader(&self) -> Option<&EntityVisual> {
        self.entities.values().find(|e| e.is_leader)
    }
    
    /// Generate formation lines for rendering.
    /// Returns pairs of (start, end) positions for formation connectors.
    pub fn get_formation_lines(&self) -> Vec<(Vec3, Vec3)> {
        let mut lines = Vec::new();
        
        if let Some(leader) = self.get_leader() {
            for entity in self.entities.values() {
                if !entity.is_leader && entity.id != leader.id {
                    lines.push((leader.position, entity.position));
                }
            }
        }
        
        lines
    }
    
    /// Get count of alive entities.
    pub fn alive_count(&self) -> usize {
        self.entities.len()
    }
}

// =============================================================================
// MESH GENERATION
// =============================================================================

/// Vertex for swarm entity mesh.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SwarmVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Generate instance data for GPU instancing.
pub struct SwarmInstance {
    pub transform: [[f32; 4]; 4],  // 4x4 matrix
    pub color: [f32; 4],
    pub heat: f32,
    pub _padding: [f32; 3],
}

impl SwarmInstance {
    pub fn from_visual(visual: &EntityVisual) -> Self {
        // Build transform matrix
        let scale_mat = glam::Mat4::from_scale(visual.scale);
        let rot_mat = glam::Mat4::from_quat(visual.orientation);
        let trans_mat = glam::Mat4::from_translation(visual.position);
        let transform = trans_mat * rot_mat * scale_mat;
        
        Self {
            transform: transform.to_cols_array_2d(),
            color: visual.color.to_array(),
            heat: visual.heat,
            _padding: [0.0; 3],
        }
    }
}

/// Generate a simple arrow/delta mesh for entities.
pub fn generate_entity_mesh() -> (Vec<SwarmVertex>, Vec<u32>) {
    // Simple delta wing shape
    let vertices = vec![
        // Nose
        SwarmVertex {
            position: [0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        },
        // Left wing tip
        SwarmVertex {
            position: [-0.5, 0.0, -0.5],
            normal: [0.0, 1.0, 0.0],
            color: [0.8, 0.8, 0.8, 1.0],
        },
        // Right wing tip
        SwarmVertex {
            position: [0.5, 0.0, -0.5],
            normal: [0.0, 1.0, 0.0],
            color: [0.8, 0.8, 0.8, 1.0],
        },
        // Center back
        SwarmVertex {
            position: [0.0, 0.0, -0.3],
            normal: [0.0, 1.0, 0.0],
            color: [0.6, 0.6, 0.6, 1.0],
        },
    ];
    
    let indices = vec![
        0, 1, 3,  // Left side
        0, 3, 2,  // Right side
    ];
    
    (vertices, indices)
}

/// Generate trail mesh for an entity.
pub fn generate_trail_mesh(
    trail: &EntityTrail,
    color: Vec4,
    width: f32,
) -> (Vec<SwarmVertex>, Vec<u32>) {
    let trail_data = trail.get_trail_with_alpha();
    
    if trail_data.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    
    let mut vertices = Vec::with_capacity(trail_data.len() * 2);
    let mut indices = Vec::new();
    
    for (i, (pos, alpha)) in trail_data.iter().enumerate() {
        // Get direction for perpendicular offset
        let dir = if i < trail_data.len() - 1 {
            (trail_data[i + 1].0 - *pos).normalize()
        } else {
            (*pos - trail_data[i - 1].0).normalize()
        };
        
        // Perpendicular in XZ plane (assuming Y is up)
        let perp = Vec3::new(-dir.z, 0.0, dir.x) * width * 0.5;
        
        let trail_color = [color.x, color.y, color.z, color.w * alpha];
        
        // Left vertex
        vertices.push(SwarmVertex {
            position: (*pos - perp).to_array(),
            normal: [0.0, 1.0, 0.0],
            color: trail_color,
        });
        
        // Right vertex
        vertices.push(SwarmVertex {
            position: (*pos + perp).to_array(),
            normal: [0.0, 1.0, 0.0],
            color: trail_color,
        });
        
        // Add quad indices (except for last segment)
        if i < trail_data.len() - 1 {
            let base = (i * 2) as u32;
            indices.extend_from_slice(&[
                base, base + 1, base + 2,
                base + 1, base + 3, base + 2,
            ]);
        }
    }
    
    (vertices, indices)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ontic_bridge::swarm::SwarmData;
    
    #[test]
    fn test_swarm_renderer_update() {
        let config = SwarmRenderConfig::default();
        let mut renderer = SwarmRenderer::new(config);
        
        // Create test swarm
        let entities = vec![
            EntityState::new(1, EntityType::BlueForce),
            EntityState::new(2, EntityType::BlueForce),
        ];
        let swarm = SwarmData::new(entities, 1000);
        
        renderer.update(&swarm);
        
        assert_eq!(renderer.alive_count(), 2);
        assert!(renderer.get_entity(1).is_some());
        assert!(renderer.get_entity(2).is_some());
    }
    
    #[test]
    fn test_entity_mesh_generation() {
        let (vertices, indices) = generate_entity_mesh();
        assert_eq!(vertices.len(), 4);
        assert_eq!(indices.len(), 6);
    }
}

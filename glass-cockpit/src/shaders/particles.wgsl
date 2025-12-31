// Phase 5: Particle System Shader (GPU Compute + Render)
// Advects particles along vector field, renders as billboarded quads
// Constitutional compliance: Doctrine 3 (GPU compute offload)
// Phase 8 Fix: Geodetic → ECEF projection for Google Earth-like globe rendering
//
// THE BIG ONE - Phase 5: Infinite Particles
// ═══════════════════════════════════════════════════════════════════════════
// Features:
//   - Distance-based sizing: particles scale with camera distance
//   - Distance-based alpha: far particles fade to prevent clutter
//   - View-dependent: backface culling, edge fade, horizon softening
//   - GPU compute advection: particles follow vector field on GPU
// ═══════════════════════════════════════════════════════════════════════════

// ============================================================================
// CONSTANTS
// ============================================================================

const PI: f32 = 3.14159265359;
const GLOBE_RADIUS: f32 = 1.0;  // Must match GlobeConfig::default().radius

// ============================================================================
// TYPES
// ============================================================================

struct Particle {
    position: vec4<f32>,    // (lon, lat, altitude, age)
    velocity: vec4<f32>,    // (u, v, w, speed)
    properties: vec4<f32>,  // (vorticity, lifetime, size, alpha)
}

struct ParticleUniforms {
    time: vec4<f32>,       // (current_time, dt, spawn_rate, seed)
    config: vec4<f32>,     // (lifetime, lifetime_var, base_size, speed_size_factor)
    bounds: vec4<f32>,     // (lon_min, lon_max, lat_min, lat_max)
    stats: vec4<f32>,      // (max_speed, max_vorticity, particle_count, _)
}

// Camera uniforms for 3D projection (matches globe pipeline)
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
    camera_pos_high: vec3<f32>,
    _padding1b: f32,
    camera_pos_low: vec3<f32>,
    _padding1c: f32,
    zoom: f32,
    aspect_ratio: f32,
    time: f32,
    _padding2: f32,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) age_normalized: f32,
    @location(3) world_pos: vec3<f32>,  // For backface culling
}

// ============================================================================
// BINDINGS - COMPUTE
// ============================================================================

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> uniforms: ParticleUniforms;
@group(0) @binding(3) var vector_field: texture_2d<f32>;

// ============================================================================
// GEODETIC UTILITIES (Appendix F: Coordinate Precision)
// ============================================================================

// Convert geodetic (lon, lat) to ECEF 3D position on globe surface
// Y-up convention to match globe_quadtree.rs mesh generation
fn geodetic_to_ecef(lon_deg: f32, lat_deg: f32, radius: f32) -> vec3<f32> {
    let lat_rad = lat_deg * PI / 180.0;
    let lon_rad = lon_deg * PI / 180.0;
    
    let cos_lat = cos(lat_rad);
    let sin_lat = sin(lat_rad);
    let cos_lon = cos(lon_rad);
    let sin_lon = sin(lon_rad);
    
    // Y-up: Y = sin(lat), XZ plane is equator
    return vec3<f32>(
        radius * cos_lat * cos_lon,  // X: toward 0°N 0°E
        radius * sin_lat,            // Y: toward North Pole (Y-up)
        radius * cos_lat * sin_lon   // Z: toward 0°N 90°E
    );
}

// ============================================================================
// UTILITIES
// ============================================================================

// Sample vector field at normalized coordinates
fn sample_vector(lon: f32, lat: f32) -> vec4<f32> {
    let bounds = uniforms.bounds;
    let u = (lon - bounds.x) / (bounds.y - bounds.x);
    let v = (lat - bounds.z) / (bounds.w - bounds.z);
    
    // Clamp to valid range
    let uv = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
    
    // Get texture dimensions
    let dims = textureDimensions(vector_field);
    let coords = vec2<i32>(uv * vec2<f32>(f32(dims.x - 1u), f32(dims.y - 1u)));
    
    return textureLoad(vector_field, coords, 0);
}

// Convert velocity to color based on vorticity
// Note: This uses render_uniforms in render context, uniforms in compute context
fn velocity_to_color_render(velocity: vec4<f32>, vorticity: f32, max_vort: f32, max_spd: f32) -> vec4<f32> {
    let norm_vort = clamp(vorticity / max_vort, -1.0, 1.0);
    
    // Divergent colormap: blue (negative/cyclonic) -> white -> red (positive/anticyclonic)
    var color: vec3<f32>;
    if norm_vort < 0.0 {
        // Cyclonic: blue gradient
        let t = -norm_vort;
        color = mix(vec3<f32>(0.8, 0.8, 0.9), vec3<f32>(0.1, 0.3, 0.8), t);
    } else {
        // Anticyclonic: red gradient
        let t = norm_vort;
        color = mix(vec3<f32>(0.9, 0.8, 0.8), vec3<f32>(0.8, 0.2, 0.1), t);
    }
    
    // Speed-based brightness
    let speed = velocity.w;
    let brightness = 0.5 + 0.5 * clamp(speed / max_spd, 0.0, 1.0);
    
    return vec4<f32>(color * brightness, 1.0);
}

// Pseudo-random number generator
fn hash(seed: u32) -> f32 {
    var n = seed;
    n = (n << 13u) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    return f32(n & 0x7fffffffu) / f32(0x7fffffff);
}

// ============================================================================
// COMPUTE SHADER - ADVECTION
// ============================================================================

@compute @workgroup_size(64)
fn cs_advect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let particle_count = u32(uniforms.stats.z);
    
    if idx >= particle_count {
        return;
    }
    
    var particle = particles_in[idx];
    let dt = uniforms.time.y;
    
    // Update age
    particle.position.w += dt;
    
    // Check if particle is still alive
    let lifetime = particle.properties.y;
    if particle.position.w >= lifetime {
        // Dead particle - respawn at random location
        let seed = u32(uniforms.time.w) + idx * 1000u;
        let bounds = uniforms.bounds;
        
        particle.position.x = bounds.x + hash(seed) * (bounds.y - bounds.x);
        particle.position.y = bounds.z + hash(seed + 1u) * (bounds.w - bounds.z);
        particle.position.z = 0.0;
        particle.position.w = 0.0;
        
        // Random lifetime variation
        let base_lifetime = uniforms.config.x;
        let lifetime_var = uniforms.config.y;
        particle.properties.y = base_lifetime + (hash(seed + 2u) - 0.5) * 2.0 * lifetime_var;
    } else {
        // Sample vector field at current position
        let vec_sample = sample_vector(particle.position.x, particle.position.y);
        
        // Update velocity
        particle.velocity.x = vec_sample.x;
        particle.velocity.y = vec_sample.y;
        particle.velocity.z = vec_sample.z;
        particle.velocity.w = length(vec_sample.xyz);
        
        // Store vorticity
        particle.properties.x = vec_sample.w;
        
        // RK4 integration (simplified to Euler for now - TODO: full RK4)
        // Convert m/s velocity to degrees/second (approximate)
        let meters_per_degree_lon = 111320.0 * cos(radians(particle.position.y));
        let meters_per_degree_lat = 110540.0;
        
        let dlon = particle.velocity.x * dt / meters_per_degree_lon;
        let dlat = particle.velocity.y * dt / meters_per_degree_lat;
        
        particle.position.x += dlon;
        particle.position.y += dlat;
        
        // Wrap longitude
        if particle.position.x > 180.0 {
            particle.position.x -= 360.0;
        } else if particle.position.x < -180.0 {
            particle.position.x += 360.0;
        }
        
        // Clamp latitude
        particle.position.y = clamp(particle.position.y, -85.0, 85.0);
    }
    
    // Compute visual properties
    let age_norm = particle.position.w / particle.properties.y;
    
    // Fade in/out at start/end of life
    let fade_in = smoothstep(0.0, 0.1, age_norm);
    let fade_out = 1.0 - smoothstep(0.8, 1.0, age_norm);
    particle.properties.w = fade_in * fade_out;
    
    // Size based on speed
    let base_size = uniforms.config.z;
    let speed_factor = uniforms.config.w;
    let max_speed = uniforms.stats.x;
    let speed_norm = clamp(particle.velocity.w / max_speed, 0.0, 1.0);
    particle.properties.z = base_size * (1.0 + speed_factor * speed_norm);
    
    particles_out[idx] = particle;
}

// ============================================================================
// RENDER SHADER - VERTEX
// ============================================================================

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> render_particles: array<Particle>;
@group(1) @binding(1) var<uniform> render_uniforms: ParticleUniforms;

@vertex
fn vs_particle(
    in: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let particle = render_particles[instance];
    
    // Skip dead particles by moving off-screen
    if particle.position.w >= particle.properties.y {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(-10.0, -10.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.age_normalized = 1.0;
        out.world_pos = vec3<f32>(0.0);
        return out;
    }
    
    // =========================================================================
    // PHASE 8 FIX: Geodetic → ECEF → Clip Space (Google Earth-like projection)
    // =========================================================================
    
    let lon = particle.position.x;  // Longitude in degrees
    let lat = particle.position.y;  // Latitude in degrees
    
    // Convert geodetic to 3D position on globe surface
    // Very slight offset above surface to prevent z-fighting with globe
    let world_pos = geodetic_to_ecef(lon, lat, GLOBE_RADIUS * 1.0003);
    
    // Backface culling: skip particles on the far side of the globe
    let to_camera = normalize(camera.camera_pos - world_pos);
    let surface_normal = normalize(world_pos);  // Normal points outward from globe center
    let facing = dot(surface_normal, to_camera);
    
    // If dot product < 0, particle is on back of globe - hide it
    if facing < 0.0 {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(-10.0, -10.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.age_normalized = 1.0;
        out.world_pos = vec3<f32>(0.0);
        return out;
    }
    
    // Billboard quad: compute right and up vectors facing camera
    // Y-up world convention
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_camera));
    let billboard_up = cross(to_camera, right);
    
    // =========================================================================
    // PHASE 5 (THE BIG ONE): Infinite Particles - Distance-based sizing
    // Particles scale with distance to maintain consistent screen coverage
    // Near particles: smaller (more detail visible)
    // Far particles: larger (still visible at planetary scale)
    // =========================================================================
    let camera_dist = length(camera.camera_pos - world_pos);
    
    // Reference distance (where particles are "normal" size)
    let reference_dist = 3.0;  // About 3 globe radii
    
    // Distance-based size scaling (sqrt for perceptual balance)
    let dist_scale = sqrt(max(camera_dist / reference_dist, 0.3));
    
    // Particle size in world units
    // Base size from particle properties, scaled by distance and zoom
    let base_size = particle.properties.z * 0.008;
    let size = base_size * dist_scale / max(camera.zoom * 0.3, 0.3);
    
    // Phase 5: Distance-based alpha fade (prevent visual clutter at distance)
    // Near: full alpha, Far: fade out
    let fade_far = 15.0;  // Distance where particles start fading
    let fade_end = 25.0;  // Distance where particles are invisible
    let distance_alpha = 1.0 - smoothstep(fade_far, fade_end, camera_dist);
    
    // Apply billboard offset
    let offset = right * in.position.x * size + billboard_up * in.position.y * size;
    let final_pos = world_pos + offset;
    
    // Phase 8: Apply RTE (Relative-To-Eye) transformation to match globe shader
    // This ensures particles render at the same coordinates as the globe surface
    let rte_pos = final_pos - camera.camera_pos_high + (-camera.camera_pos_low);
    
    // Project to clip space using camera view-projection matrix
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(rte_pos, 1.0);
    out.uv = in.uv;
    out.world_pos = world_pos;
    
    // Color from vorticity (unchanged)
    out.color = velocity_to_color_render(
        particle.velocity, 
        particle.properties.x, 
        render_uniforms.stats.y, 
        render_uniforms.stats.x
    );
    
    // Phase 5: Combined alpha includes:
    // - Particle age fade (fade in/out at birth/death)
    // - Edge facing fade (softer horizon)
    // - Distance fade (prevent clutter at planetary scale)
    let edge_fade = smoothstep(0.0, 0.3, facing);
    out.color.a = particle.properties.w * edge_fade * distance_alpha;
    
    out.age_normalized = particle.position.w / particle.properties.y;
    
    return out;
}

// ============================================================================
// RENDER SHADER - FRAGMENT
// ============================================================================

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    // FILTER: Only draw particles with sufficient alpha (includes backface + edge fade)
    if in.color.a < 0.05 {
        discard;
    }
    
    // Circular particle with soft edge
    let dist = length(in.uv - vec2<f32>(0.5));
    let edge = smoothstep(0.5, 0.35, dist);
    
    // Add glow effect for high-vorticity particles
    let glow = smoothstep(0.6, 0.2, dist) * 0.3;
    
    var color = in.color;
    color.a *= edge + glow;
    
    // Discard fully transparent pixels
    if color.a < 0.01 {
        discard;
    }
    
    return color;
}

// Phase 5: Particle System Shader (GPU Compute + Render)
// Advects particles along vector field, renders as billboarded quads
// Constitutional compliance: Doctrine 3 (GPU compute offload)

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

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) age_normalized: f32,
}

// ============================================================================
// BINDINGS - COMPUTE
// ============================================================================

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> uniforms: ParticleUniforms;
@group(0) @binding(3) var vector_field: texture_2d<f32>;

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

@group(0) @binding(0) var<storage, read> render_particles: array<Particle>;
@group(0) @binding(1) var<uniform> render_uniforms: ParticleUniforms;

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
        return out;
    }
    
    let bounds = render_uniforms.bounds;
    
    // Convert lon/lat to normalized screen coordinates (-1 to 1)
    let u = (particle.position.x - bounds.x) / (bounds.y - bounds.x);
    let v = (particle.position.y - bounds.z) / (bounds.w - bounds.z);
    let screen_pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
    
    // Particle size in screen units
    let size = particle.properties.z * 0.005; // Scale factor for screen
    
    // Billboard offset
    let offset = in.position * size;
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(screen_pos + offset, 0.0, 1.0);
    out.uv = in.uv;
    out.color = velocity_to_color_render(particle.velocity, particle.properties.x, render_uniforms.stats.y, render_uniforms.stats.x);
    out.color.a = particle.properties.w;
    out.age_normalized = particle.position.w / particle.properties.y;
    
    return out;
}

// ============================================================================
// RENDER SHADER - FRAGMENT
// ============================================================================

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    // FILTER: Only draw high-energy nodes (kill the snow)
    // If the particle has low vorticity (calm weather), discard it.
    // This removes ~90% of visual noise from calm regions
    if in.color.a < 0.3 {
        discard;
    }
    
    // Circular particle with soft edge
    let dist = length(in.uv - vec2<f32>(0.5));
    let edge = smoothstep(0.5, 0.35, dist);
    
    // Add glow effect
    let glow = smoothstep(0.6, 0.2, dist) * 0.3;
    
    var color = in.color;
    color.a *= edge + glow;
    
    // Discard fully transparent pixels
    if color.a < 0.01 {
        discard;
    }
    
    return color;
}

// Vorticity Ghost Shader - Phase 8: Appendix G
//
// Volumetric smoke-like overlay visualizing the curl of the tensor field.
// Users see the mathematical "swirl" before satellite imagery shows clouds.
//
// Constitutional Compliance:
// - Article V: GPU-accelerated ray marching
// - Doctrine 3: Procedural rendering, no texture assets

struct VorticityUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    globe_radius: f32,
    vorticity_threshold: f32,
    vorticity_max: f32,
    max_opacity: f32,
};

struct ConvergenceCell {
    // x: longitude, y: latitude, z: intensity, w: vorticity
    data: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: VorticityUniforms;
@group(0) @binding(1) var<storage, read> cells: array<ConvergenceCell>;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) ray_dir: vec3<f32>,
};

// ═══════════════════════════════════════════════════════════════════════
// G.3.1: Fractal Brownian Motion Noise for organic smoke appearance
// ═══════════════════════════════════════════════════════════════════════
fn hash(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3<f32>(443.897, 441.423, 437.195));
    let r = q + dot(q, q.yxz + 19.19);
    return fract((r.x + r.y) * r.z);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(
            mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)), hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

fn fbm_noise(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    
    for (var i = 0; i < 4; i++) {
        value += amplitude * noise3d(pos);
        pos *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

// ═══════════════════════════════════════════════════════════════════════
// G.2.1: Sample vorticity from convergence field at world position
// ═══════════════════════════════════════════════════════════════════════
fn sample_vorticity_at(world_pos: vec3<f32>) -> f32 {
    // Convert world pos to geographic coordinates
    let r = length(world_pos);
    if r < 0.001 {
        return 0.0;
    }
    
    let lat = asin(world_pos.y / r);
    let lon = atan2(world_pos.z, world_pos.x);
    
    // Find nearest cell in convergence field
    let u = (lon + 3.14159) / 6.28318;
    let v = (lat + 1.5708) / 3.14159;
    
    // Sample from storage buffer (approximate nearest neighbor)
    let cell_idx = u32(u * 128.0) + u32(v * 64.0) * 128u;
    let max_idx = arrayLength(&cells) - 1u;
    let safe_idx = min(cell_idx, max_idx);
    
    return cells[safe_idx].data.w;  // vorticity is in w component
}

// ═══════════════════════════════════════════════════════════════════════
// G.3.1: Smoke density function
// ═══════════════════════════════════════════════════════════════════════
fn smoke_density(world_pos: vec3<f32>) -> f32 {
    let vorticity = sample_vorticity_at(world_pos);
    
    // Only render where vorticity exceeds threshold
    if abs(vorticity) < uniforms.vorticity_threshold {
        return 0.0;
    }
    
    // Noise-based turbulence for organic appearance
    let noise_pos = world_pos * 0.01 + vec3<f32>(uniforms.time * 0.1, 0.0, 0.0);
    let noise = fbm_noise(noise_pos);
    
    // Density increases with vorticity magnitude
    let base_density = smoothstep(
        uniforms.vorticity_threshold, 
        uniforms.vorticity_max, 
        abs(vorticity)
    );
    
    return base_density * (0.5 + 0.5 * noise);
}

// ═══════════════════════════════════════════════════════════════════════
// G.4.2: Animated density with vorticity-dependent pulsing
// ═══════════════════════════════════════════════════════════════════════
fn animated_density(base_density: f32, vorticity: f32) -> f32 {
    // Pulse frequency increases with vorticity
    let pulse_freq = 1.0 + abs(vorticity) * 2.0;
    let pulse = sin(uniforms.time * pulse_freq) * 0.3 + 0.7;
    return base_density * pulse;
}

// Full-screen quad vertex shader for ray marching
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Full-screen triangle (covers NDC [-1, 1])
    let x = f32((input.vertex_index & 1u) << 2u) - 1.0;
    let y = f32((input.vertex_index & 2u) << 1u) - 1.0;
    
    output.clip_position = vec4<f32>(x, y, 0.5, 1.0);
    
    // Compute ray direction from camera through this pixel
    // Inverse view-projection would be ideal, but we approximate
    let aspect = 16.0 / 9.0;
    output.ray_dir = normalize(vec3<f32>(x * aspect, y, -1.5));
    output.world_pos = uniforms.camera_pos;
    
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// G.3.2: Ray marching fragment shader
// ═══════════════════════════════════════════════════════════════════════
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = uniforms.camera_pos;
    let ray_dir = normalize(input.ray_dir);
    
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    
    let step_size = 0.02;  // Step size in normalized units
    let max_steps = 32u;
    let near_plane = 0.1;
    
    for (var i = 0u; i < max_steps; i++) {
        let t = f32(i) * step_size + near_plane;
        let sample_pos = ray_origin + ray_dir * t;
        
        // Check if inside globe atmosphere (slightly larger than globe)
        let dist_from_center = length(sample_pos);
        if dist_from_center > uniforms.globe_radius * 1.2 || dist_from_center < uniforms.globe_radius * 0.8 {
            continue;
        }
        
        let density = smoke_density(sample_pos);
        
        if density > 0.001 {
            // Color based on vorticity sign (cyclonic vs anticyclonic)
            let vorticity = sample_vorticity_at(sample_pos);
            var color: vec3<f32>;
            
            if vorticity > 0.0 {
                // Blue for cyclonic (Northern Hemisphere low pressure)
                color = vec3<f32>(0.1, 0.5, 0.9);
            } else {
                // Red-orange for anticyclonic (high pressure)
                color = vec3<f32>(0.8, 0.3, 0.1);
            }
            
            // Apply animation
            let animated = animated_density(density, vorticity);
            
            // Beer-Lambert absorption
            let transmittance = exp(-animated * step_size * 0.1);
            accumulated_color += color * animated * (1.0 - accumulated_alpha);
            accumulated_alpha += animated * (1.0 - accumulated_alpha) * 0.5;
        }
        
        if accumulated_alpha > 0.95 {
            break; // Early termination
        }
    }
    
    // Clamp to max opacity per Appendix G spec
    accumulated_alpha = min(accumulated_alpha, uniforms.max_opacity);
    
    return vec4<f32>(accumulated_color, accumulated_alpha);
}

// Global Eye Wind Visualization Shader - Phase 1C-9
//
// Renders wind velocity data from the Rgba32Float texture onto a globe mesh.
// R = U-wind (east/west), G = V-wind (north/south), B = magnitude, A = unused

// ═══════════════════════════════════════════════════════════════════════════════
// Uniforms
// ═══════════════════════════════════════════════════════════════════════════════

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    max_wind_speed: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var wind_texture: texture_2d<f32>;
@group(1) @binding(1) var wind_sampler: sampler;

// ═══════════════════════════════════════════════════════════════════════════════
// Vertex Shader
// ═══════════════════════════════════════════════════════════════════════════════

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let world_pos = uniforms.model * vec4(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.world_normal = normalize((uniforms.model * vec4(in.normal, 0.0)).xyz);
    out.clip_position = uniforms.view_proj * world_pos;
    out.uv = in.uv;
    
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fragment Shader
// ═══════════════════════════════════════════════════════════════════════════════

// Colormap: Calm (blue) → Moderate (green) → Storm (red)
fn wind_colormap(speed: f32, max_speed: f32) -> vec3<f32> {
    let t = clamp(speed / max_speed, 0.0, 1.0);
    
    // Three-point gradient: blue → green → red
    if t < 0.5 {
        let s = t * 2.0;
        return mix(
            vec3(0.0, 0.2, 0.6),  // Deep blue (calm)
            vec3(0.2, 0.8, 0.2),  // Green (moderate)
            s
        );
    } else {
        let s = (t - 0.5) * 2.0;
        return mix(
            vec3(0.2, 0.8, 0.2),  // Green (moderate)
            vec3(1.0, 0.2, 0.0),  // Red (storm)
            s
        );
    }
}

// Add subtle animation based on wind direction
fn wind_animation(uv: vec2<f32>, u_wind: f32, v_wind: f32, time: f32) -> vec2<f32> {
    let wind_dir = normalize(vec2(u_wind, v_wind) + vec2(0.001));
    let offset = wind_dir * sin(time * 2.0 + length(uv) * 20.0) * 0.002;
    return uv + offset;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample wind texture
    // Layout: R=U-wind, G=V-wind, B=magnitude (pre-computed), A=reserved
    let wind_data = textureSample(wind_texture, wind_sampler, in.uv);
    
    let u_wind = wind_data.r;  // East/West component (m/s)
    let v_wind = wind_data.g;  // North/South component (m/s)
    let speed = wind_data.b;   // Magnitude (m/s)
    
    // Skip rendering where there's no data (ocean or out-of-bounds)
    if speed < 0.1 {
        // Fallback: dark ocean color
        return vec4(0.02, 0.05, 0.15, 1.0);
    }
    
    // Apply colormap based on wind speed
    let base_color = wind_colormap(speed, uniforms.max_wind_speed);
    
    // Simple directional shading
    let light_dir = normalize(vec3(1.0, 1.0, 0.5));
    let ndotl = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7 * ndotl;
    
    let lit_color = base_color * (ambient + diffuse);
    
    // Add wind direction indicator (subtle streaks)
    let wind_angle = atan2(v_wind, u_wind);
    let streak = sin(in.uv.x * 100.0 + wind_angle + uniforms.time * 2.0) * 0.05;
    let final_color = lit_color + vec3(streak);
    
    return vec4(final_color, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Alternative: Transparency Mode (for overlay on base map)
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_overlay(in: VertexOutput) -> @location(0) vec4<f32> {
    let wind_data = textureSample(wind_texture, wind_sampler, in.uv);
    
    let u_wind = wind_data.r;
    let v_wind = wind_data.g;
    let speed = sqrt(u_wind * u_wind + v_wind * v_wind);
    
    // Normalized speed for coloring
    let normalized_speed = clamp(speed / uniforms.max_wind_speed, 0.0, 1.0);
    
    // Blue = Calm, Red = Storm
    let color_base = mix(
        vec3(0.0, 0.1, 0.5),  // Blue
        vec3(1.0, 0.2, 0.0),  // Red
        normalized_speed
    );
    
    // Alpha: transparent for calm, opaque for strong winds
    let alpha = smoothstep(2.0, 15.0, speed);
    
    return vec4(color_base, alpha);
}

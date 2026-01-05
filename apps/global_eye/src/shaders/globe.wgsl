// Global Eye Wind Visualization Shader - Phase 1C-10
//
// Renders wind velocity data from the Rgba32Float texture onto a globe mesh.
// R = U-wind (east/west), G = V-wind (north/south), B = magnitude, A = unused

// ═══════════════════════════════════════════════════════════════════════════════
// Uniforms
// ═══════════════════════════════════════════════════════════════════════════════

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var wind_texture: texture_2d<f32>;
@group(1) @binding(1) var wind_sampler: sampler;

// Constants
const MAX_WIND_SPEED: f32 = 30.0;  // m/s - typical max for visualization

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
    
    // Model matrix is identity (globe at origin)
    out.world_position = in.position;
    out.world_normal = in.normal;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    out.uv = in.uv;
    
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fragment Shader
// ═══════════════════════════════════════════════════════════════════════════════

// Colormap: Calm (blue) → Moderate (cyan) → Strong (green) → Storm (red)
fn wind_colormap(speed: f32) -> vec3<f32> {
    let t = clamp(speed / MAX_WIND_SPEED, 0.0, 1.0);
    
    // Four-point gradient for better visual range
    if t < 0.33 {
        let s = t * 3.0;
        return mix(
            vec3(0.1, 0.2, 0.5),  // Deep blue (calm)
            vec3(0.1, 0.5, 0.7),  // Cyan (light breeze)
            s
        );
    } else if t < 0.66 {
        let s = (t - 0.33) * 3.0;
        return mix(
            vec3(0.1, 0.5, 0.7),  // Cyan
            vec3(0.3, 0.8, 0.3),  // Green (moderate)
            s
        );
    } else {
        let s = (t - 0.66) * 3.0;
        return mix(
            vec3(0.3, 0.8, 0.3),  // Green
            vec3(1.0, 0.3, 0.1),  // Red (storm)
            s
        );
    }
}

// Ocean color based on latitude
fn ocean_color(uv: vec2<f32>) -> vec3<f32> {
    // Darker near poles, brighter near equator
    let lat_factor = abs(uv.y - 0.5) * 2.0;
    return mix(
        vec3(0.02, 0.08, 0.20),  // Tropical ocean
        vec3(0.01, 0.03, 0.08),  // Polar ocean
        lat_factor
    );
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
    let speed = wind_data.b;   // Pre-computed magnitude (m/s)
    
    // Compute actual speed if magnitude wasn't pre-computed
    let actual_speed = select(sqrt(u_wind * u_wind + v_wind * v_wind), speed, speed > 0.01);
    
    // Base color
    var base_color: vec3<f32>;
    
    // Check for valid wind data
    if actual_speed < 0.1 && abs(u_wind) < 0.01 && abs(v_wind) < 0.01 {
        // No data - ocean/void
        base_color = ocean_color(in.uv);
    } else {
        // Has wind data - apply colormap
        base_color = wind_colormap(actual_speed);
    }
    
    // Simple atmospheric shading
    let view_dir = normalize(camera.camera_pos - in.world_position);
    let fresnel = pow(1.0 - max(dot(view_dir, in.world_normal), 0.0), 2.0);
    let atmosphere = vec3(0.3, 0.5, 0.8) * fresnel * 0.3;
    
    // Simple directional lighting (sun from upper-right)
    let light_dir = normalize(vec3(0.5, 0.8, 0.3));
    let ndotl = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.35;
    let diffuse = 0.65 * ndotl;
    
    // Combine lighting
    let lit_color = base_color * (ambient + diffuse);
    
    // Add wind direction streaks for areas with wind
    var final_color = lit_color;
    if actual_speed > 1.0 {
        let wind_angle = atan2(v_wind, u_wind);
        // Create subtle directional streaks
        let streak_pattern = sin(
            dot(in.uv, vec2(cos(wind_angle), sin(wind_angle))) * 150.0
        );
        let streak_intensity = actual_speed / MAX_WIND_SPEED * 0.1;
        final_color += vec3(streak_pattern * streak_intensity);
    }
    
    // Add atmosphere glow at edges
    final_color += atmosphere;
    
    return vec4(final_color, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Alternative: Demo Mode (animated gradient for testing)
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_demo(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create a demo visualization without actual weather data
    // Uses UV coordinates to generate a fake wind pattern
    
    let fake_speed = sin(in.uv.x * 10.0) * cos(in.uv.y * 10.0) * 15.0 + 15.0;
    let fake_u = sin(in.uv.y * 20.0) * 10.0;
    let fake_v = cos(in.uv.x * 20.0) * 10.0;
    
    let base_color = wind_colormap(fake_speed);
    
    // Fresnel edge glow
    let view_dir = normalize(camera.camera_pos - in.world_position);
    let fresnel = pow(1.0 - max(dot(view_dir, in.world_normal), 0.0), 3.0);
    let atmosphere = vec3(0.4, 0.6, 1.0) * fresnel * 0.4;
    
    // Lighting
    let light_dir = normalize(vec3(0.5, 0.8, 0.3));
    let ndotl = max(dot(in.world_normal, light_dir), 0.0);
    let lit_color = base_color * (0.35 + 0.65 * ndotl);
    
    return vec4(lit_color + atmosphere, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Alternative: Overlay Mode (transparent for compositing)
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_overlay(in: VertexOutput) -> @location(0) vec4<f32> {
    let wind_data = textureSample(wind_texture, wind_sampler, in.uv);
    
    let u_wind = wind_data.r;
    let v_wind = wind_data.g;
    let speed = sqrt(u_wind * u_wind + v_wind * v_wind);
    
    // Color based on wind speed
    let color_base = wind_colormap(speed);
    
    // Alpha: more visible for stronger winds
    let alpha = smoothstep(1.0, 10.0, speed) * 0.8;
    
    return vec4(color_base, alpha);
}

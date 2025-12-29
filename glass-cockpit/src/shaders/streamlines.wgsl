// Phase 5: Streamline Shader
// Renders precomputed streamlines as ribbons with magnitude-based thickness
// Constitutional compliance: Doctrine 3 (GPU rendering)

// ============================================================================
// TYPES
// ============================================================================

struct StreamlineUniforms {
    bounds: vec4<f32>,     // (lon_min, lon_max, lat_min, lat_max)
    stats: vec4<f32>,      // (max_speed, max_vorticity, base_thickness, speed_factor)
    time: vec4<f32>,       // (current_time, animation_speed, _, _)
}

struct VertexInput {
    @location(0) position: vec3<f32>,    // (lon, lat, altitude)
    @location(1) tangent: vec3<f32>,     // normalized tangent
    @location(2) properties: vec4<f32>,  // (speed, vorticity, arc_length, side)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) arc_length: f32,
    @location(2) edge_dist: f32,  // Distance from edge for anti-aliasing
}

// ============================================================================
// BINDINGS
// ============================================================================

@group(0) @binding(0) var<uniform> uniforms: StreamlineUniforms;

// ============================================================================
// UTILITIES
// ============================================================================

// Speed-based colormap (blue -> cyan -> green -> yellow -> red)
fn speed_to_color(speed: f32, max_speed: f32) -> vec3<f32> {
    let t = clamp(speed / max_speed, 0.0, 1.0);
    
    // Scientific colormap (similar to viridis)
    if t < 0.25 {
        let s = t / 0.25;
        return mix(vec3<f32>(0.267, 0.004, 0.329), vec3<f32>(0.282, 0.140, 0.458), s);
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        return mix(vec3<f32>(0.282, 0.140, 0.458), vec3<f32>(0.127, 0.566, 0.550), s);
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        return mix(vec3<f32>(0.127, 0.566, 0.550), vec3<f32>(0.741, 0.873, 0.150), s);
    } else {
        let s = (t - 0.75) / 0.25;
        return mix(vec3<f32>(0.741, 0.873, 0.150), vec3<f32>(0.993, 0.906, 0.144), s);
    }
}

// Vorticity overlay (cyclonic vs anticyclonic)
fn vorticity_overlay(base_color: vec3<f32>, vorticity: f32, max_vorticity: f32) -> vec3<f32> {
    let norm_vort = clamp(vorticity / max_vorticity, -1.0, 1.0);
    
    if abs(norm_vort) < 0.1 {
        return base_color;
    }
    
    let intensity = (abs(norm_vort) - 0.1) / 0.9;
    
    if norm_vort < 0.0 {
        // Cyclonic: blue tint
        return mix(base_color, vec3<f32>(0.3, 0.5, 0.9), intensity * 0.4);
    } else {
        // Anticyclonic: orange tint
        return mix(base_color, vec3<f32>(0.9, 0.5, 0.3), intensity * 0.4);
    }
}

// ============================================================================
// VERTEX SHADER
// ============================================================================

@vertex
fn vs_streamline(in: VertexInput) -> VertexOutput {
    let bounds = uniforms.bounds;
    let stats = uniforms.stats;
    
    // Convert lon/lat to normalized screen coordinates
    let u = (in.position.x - bounds.x) / (bounds.y - bounds.x);
    let v = (in.position.y - bounds.z) / (bounds.w - bounds.z);
    let screen_pos = vec2<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0);
    
    // Calculate ribbon width based on speed
    let base_thickness = stats.z;
    let speed_factor = stats.w;
    let max_speed = stats.x;
    let speed_norm = clamp(in.properties.x / max_speed, 0.0, 1.0);
    let thickness = base_thickness * (1.0 + speed_factor * speed_norm) * 0.005; // Screen units
    
    // Perpendicular to tangent for ribbon expansion
    let perp = vec2<f32>(-in.tangent.y, in.tangent.x);
    let offset = perp * thickness * in.properties.w; // side: -1 or 1
    
    // Animated dash pattern
    let time = uniforms.time.x;
    let anim_speed = uniforms.time.y;
    let arc_offset = in.properties.z - time * anim_speed * 0.0001;
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(screen_pos + offset, 0.0, 1.0);
    
    // Color based on speed with vorticity overlay
    let base_color = speed_to_color(in.properties.x, max_speed);
    let color = vorticity_overlay(base_color, in.properties.y, stats.y);
    out.color = vec4<f32>(color, 0.85);
    
    out.arc_length = arc_offset;
    out.edge_dist = in.properties.w; // -1 to 1 across ribbon width
    
    return out;
}

// ============================================================================
// FRAGMENT SHADER
// ============================================================================

@fragment
fn fs_streamline(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color;
    
    // MASK: Sphere Projection - constrain streamlines to globe bounds
    // Calculate distance from center of screen (0.0, 0.0 in NDC)
    let screen_center = in.clip_position.xy / in.clip_position.w;
    let dist_from_center = length(screen_center);
    
    // Hard cut at the globe edge (fade out at NDC radius ~0.85)
    let alpha_mask = 1.0 - smoothstep(0.75, 0.90, dist_from_center);
    color.a *= alpha_mask;
    
    // Anti-aliased edge
    let edge_fade = 1.0 - smoothstep(0.7, 1.0, abs(in.edge_dist));
    color.a *= edge_fade;
    
    // Optional animated dash pattern (commented out for solid lines)
    // let dash_phase = fract(in.arc_length * 0.00001);
    // if dash_phase > 0.5 {
    //     color.a *= 0.3;
    // }
    
    // Subtle center highlight
    let center_highlight = 1.0 - abs(in.edge_dist);
    let highlight = vec3<f32>(0.1) * center_highlight * center_highlight;
    color = vec4<f32>(color.r + highlight.x, color.g + highlight.y, color.b + highlight.z, color.a);
    
    if color.a < 0.01 {
        discard;
    }
    
    return color;
}

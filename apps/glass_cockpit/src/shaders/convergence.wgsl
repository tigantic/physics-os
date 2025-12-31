// Convergence Heatmap Shader
//
// Phase 6: Probabilistic convergence zone visualization
// Phase 8: Appendix D hover feedback integration
//
// Renders probability field as multi-layer heatmap with:
// - Spectral/plasma color scale
// - Intensity pulsing for high-probability zones
// - Smooth interpolation between grid cells
// - Alpha blending for compositing over globe/vectors
// - Hover state visual feedback (constriction, pulse increase)

struct ConvergenceUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,      // Camera position for RTE (xyz used, w unused)
    globe_radius: f32,
    time: f32,
    visibility_threshold: f32,
    high_intensity_threshold: f32,
    pulse_frequency: f32,
    max_intensity: f32,
    // Phase 8: Appendix D - Hover state
    hover_pos: vec2<f32>,       // lon, lat in radians
    _padding_a: f32,
    hover_intensity: f32,       // 0 = not hovering, >0 = hovering
    // Phase 8: Appendix D - Ghost mode (causal trace)
    ghost_mode: f32,            // 0 = normal, 1 = ghost/causal trace
    _pad1: f32,                 // Explicit padding for vec2 alignment
    ghost_selected_pos: vec2<f32>,  // Selected node position for ghost mode
    _pad2: vec2<f32>,           // Explicit padding for vec4 alignment
    _padding: vec4<f32>,        // Padding to 160 bytes total
};

struct ConvergenceCell {
    // x: longitude, y: latitude, z: intensity, w: vorticity
    data: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: ConvergenceUniforms;
@group(0) @binding(1) var<storage, read> cells: array<ConvergenceCell>;
@group(0) @binding(2) var<uniform> grid_dims: vec2<u32>;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) intensity: f32,
    @location(1) vorticity: f32,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

// Convert geographic coordinates to 3D globe position
fn geo_to_globe(lon: f32, lat: f32, radius: f32) -> vec3<f32> {
    return vec3<f32>(
        radius * cos(lat) * cos(lon),
        radius * sin(lat),
        radius * cos(lat) * sin(lon)
    );
}

// Plasma colormap (perceptually uniform, good for heatmaps)
fn plasma_colormap(t: f32) -> vec3<f32> {
    // Attempt to approximate matplotlib plasma colormap
    let c0 = vec3<f32>(0.050383, 0.029803, 0.527975);
    let c1 = vec3<f32>(0.494877, 0.011615, 0.657865);
    let c2 = vec3<f32>(0.798216, 0.280197, 0.469538);
    let c3 = vec3<f32>(0.973416, 0.585761, 0.254154);
    let c4 = vec3<f32>(0.940015, 0.975158, 0.131326);

    let tc = clamp(t, 0.0, 1.0);
    
    if (tc < 0.25) {
        return mix(c0, c1, tc * 4.0);
    } else if (tc < 0.5) {
        return mix(c1, c2, (tc - 0.25) * 4.0);
    } else if (tc < 0.75) {
        return mix(c2, c3, (tc - 0.5) * 4.0);
    } else {
        return mix(c3, c4, (tc - 0.75) * 4.0);
    }
}

// Inferno colormap (alternative, more red-hot feel)
fn inferno_colormap(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.001462, 0.000466, 0.013866);
    let c1 = vec3<f32>(0.341500, 0.062325, 0.429425);
    let c2 = vec3<f32>(0.735683, 0.215906, 0.330245);
    let c3 = vec3<f32>(0.978422, 0.557937, 0.034931);
    let c4 = vec3<f32>(0.988362, 0.998364, 0.644924);

    let tc = clamp(t, 0.0, 1.0);
    
    if (tc < 0.25) {
        return mix(c0, c1, tc * 4.0);
    } else if (tc < 0.5) {
        return mix(c1, c2, (tc - 0.25) * 4.0);
    } else if (tc < 0.75) {
        return mix(c2, c3, (tc - 0.5) * 4.0);
    } else {
        return mix(c3, c4, (tc - 0.75) * 4.0);
    }
}

// Combined colormap with vorticity influence
fn convergence_colormap(intensity: f32, vorticity: f32) -> vec3<f32> {
    // Base color from plasma
    var color = plasma_colormap(intensity);
    
    // Add vorticity tint (cyclonic = blue shift, anticyclonic = red shift)
    let vort_contrib = clamp(abs(vorticity), 0.0, 1.0) * 0.3;
    if (vorticity > 0.0) {
        // Cyclonic: shift toward blue
        color = mix(color, vec3<f32>(0.2, 0.4, 0.9), vort_contrib);
    } else {
        // Anticyclonic: shift toward red
        color = mix(color, vec3<f32>(0.9, 0.2, 0.3), vort_contrib);
    }
    
    return color;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    let cell = cells[input.instance_index];
    let lon = cell.data.x;
    let lat = cell.data.y;
    let intensity = cell.data.z;
    let vorticity = cell.data.w;
    
    // Skip cells below visibility threshold
    if (intensity < uniforms.visibility_threshold) {
        output.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0); // Behind camera
        return output;
    }
    
    // Quad vertices (billboard on globe surface)
    // Using switch to avoid dynamic array indexing (WGSL limitation)
    let quad_size = 0.03; // Size in radians
    var offset: vec2<f32>;
    switch (input.vertex_index) {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }
        case 1u: { offset = vec2<f32>( 1.0, -1.0); }
        case 2u: { offset = vec2<f32>( 1.0,  1.0); }
        case 3u: { offset = vec2<f32>(-1.0, -1.0); }
        case 4u: { offset = vec2<f32>( 1.0,  1.0); }
        case 5u: { offset = vec2<f32>(-1.0,  1.0); }
        default: { offset = vec2<f32>(0.0, 0.0); }
    }
    
    let adjusted_lon = lon + offset.x * quad_size;
    let adjusted_lat = lat + offset.y * quad_size;
    
    // Slight elevation above globe for visibility
    let elevation = uniforms.globe_radius * 1.002 + intensity * 0.02;
    let world_pos = geo_to_globe(adjusted_lon, adjusted_lat, elevation);
    
    // Apply RTE (Relative-To-Eye) transformation like globe.wgsl
    // This ensures heatmap renders at same scale as globe
    let rte_position = world_pos - uniforms.camera_pos.xyz;
    
    output.clip_position = uniforms.view_proj * vec4<f32>(rte_position, 1.0);
    output.intensity = intensity;
    output.vorticity = vorticity;
    output.world_pos = world_pos;
    output.uv = offset * 0.5 + 0.5;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = input.intensity;
    let vorticity = input.vorticity;
    
    // Discard if below threshold (safety check)
    if (intensity < uniforms.visibility_threshold) {
        discard;
    }
    
    // Base color from convergence colormap
    var color = convergence_colormap(intensity, vorticity);
    
    // Radial falloff for soft edges
    let center_dist = length(input.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let radial_falloff = 1.0 - smoothstep(0.6, 1.0, center_dist);
    
    // Intensity-based alpha (more intense = more opaque)
    var alpha = intensity * radial_falloff * 0.7;
    
    // Pulse animation for high-intensity zones
    if (intensity > uniforms.high_intensity_threshold) {
        let pulse = sin(uniforms.time * uniforms.pulse_frequency * 3.14159) * 0.5 + 0.5;
        let pulse_strength = (intensity - uniforms.high_intensity_threshold) / 
                            (1.0 - uniforms.high_intensity_threshold);
        
        // Brighten color during pulse
        color = mix(color, vec3<f32>(1.0, 1.0, 1.0), pulse * pulse_strength * 0.3);
        
        // Increase alpha during pulse
        alpha = alpha + pulse * pulse_strength * 0.2;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // Phase 8: Appendix D - Hover feedback
    // Constriction effect + increased pulse when hovering over convergence zone
    // ═══════════════════════════════════════════════════════════════════
    if (uniforms.hover_intensity > 0.0) {
        // Calculate distance from hover position (in geographic space)
        let cell = cells[0];  // Get current cell for position reference
        let cell_lon = cell.data.x;
        let cell_lat = cell.data.y;
        let hover_dist = length(vec2<f32>(cell_lon, cell_lat) - uniforms.hover_pos);
        
        // Constriction radius that decreases as hover intensifies
        let constriction_radius = mix(0.5, 0.1, uniforms.hover_intensity);
        let hover_factor = 1.0 - smoothstep(0.0, constriction_radius, hover_dist);
        
        // Increased pulse frequency near cursor (per D.3.2: 1.0 → 3.0)
        let hover_pulse_freq = uniforms.pulse_frequency * mix(1.0, 3.0, hover_factor);
        let hover_pulse = sin(uniforms.time * hover_pulse_freq * 3.14159) * 0.5 + 0.5;
        
        // Edge glow intensifies near cursor (per D.3.2: 0.0 → 0.8)
        let hover_glow = hover_factor * 0.8;
        color = mix(color, vec3<f32>(0.0, 0.8, 1.0), hover_glow * hover_pulse);
        alpha = alpha + hover_factor * 0.3;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // Phase 8: Appendix D - Causal Trace Ghost Mode
    // When in ghost mode, fade non-upstream nodes and highlight selected
    // Per D.3.5: Non-selected fades to 20% opacity, upstream nodes pulse
    // ═══════════════════════════════════════════════════════════════════
    if (uniforms.ghost_mode > 0.5) {
        // Calculate distance from selected node
        let selected_dist = length(input.uv - uniforms.ghost_selected_pos * 0.5 + 0.5);
        
        // Determine if this is an "upstream" node based on proximity and vorticity direction
        // Upstream nodes are neighbors that have vorticity pointing toward selected node
        let is_upstream = input.vorticity > 0.2 && selected_dist < 0.3;
        let is_selected = selected_dist < 0.05;
        
        if (is_selected) {
            // Selected node: bright pulse
            let select_pulse = sin(uniforms.time * 4.0) * 0.3 + 0.7;
            color = vec3<f32>(1.0, 0.8, 0.0) * select_pulse;
            alpha = 1.0;
        } else if (is_upstream) {
            // Upstream nodes: cyan pulse highlight
            let upstream_pulse = sin(uniforms.time * 4.0) * 0.3 + 0.7;
            color = color + vec3<f32>(0.0, 0.6, 1.0) * upstream_pulse;
            alpha = min(alpha * 1.2, 0.9);
        } else {
            // Non-relevant nodes: fade to 20% opacity, desaturate
            color = mix(color, vec3<f32>(0.1, 0.1, 0.12), 0.8);
            alpha = alpha * 0.2;
        }
    }
    
    // Edge glow for visibility
    let edge_glow = smoothstep(0.4, 0.6, center_dist) * (1.0 - smoothstep(0.8, 1.0, center_dist));
    let edge_color = color * 1.5;
    color = mix(color, edge_color, edge_glow * 0.5);
    
    // Final alpha clamping
    alpha = clamp(alpha, 0.0, 0.85);
    
    return vec4<f32>(color, alpha);
}

// Alternative fragment shader for dense heatmap rendering (full-screen quad approach)
// Use this for lower cell counts with smooth interpolation

struct HeatmapVertex {
    @builtin(vertex_index) vertex_index: u32,
};

struct HeatmapOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_heatmap(input: HeatmapVertex) -> HeatmapOutput {
    var output: HeatmapOutput;
    
    // Full-screen triangle
    let x = f32((input.vertex_index & 1u) << 2u) - 1.0;
    let y = f32((input.vertex_index & 2u) << 1u) - 1.0;
    
    output.clip_position = vec4<f32>(x, y, 0.5, 1.0);
    output.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    
    return output;
}

// Texture-based heatmap for smooth rendering
@group(1) @binding(0) var heatmap_texture: texture_2d<f32>;
@group(1) @binding(1) var heatmap_sampler: sampler;

@fragment
fn fs_heatmap(input: HeatmapOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(heatmap_texture, heatmap_sampler, input.uv);
    let intensity = sample.r;
    let vorticity = sample.g;
    
    if (intensity < uniforms.visibility_threshold) {
        discard;
    }
    
    let color = convergence_colormap(intensity, vorticity);
    
    // Pulse for high intensity
    var alpha = intensity * 0.6;
    if (intensity > uniforms.high_intensity_threshold) {
        let pulse = sin(uniforms.time * uniforms.pulse_frequency * 3.14159) * 0.5 + 0.5;
        alpha = alpha + pulse * 0.2;
    }
    
    return vec4<f32>(color, clamp(alpha, 0.0, 0.8));
}

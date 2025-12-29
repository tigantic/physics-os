// Phase 1: Procedural Grid Shader
// 
// Infinite grid using derivative-based antialiasing
// Doctrine 3: Procedural Rendering

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) near_point: vec3<f32>,
    @location(1) far_point: vec3<f32>,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Generate full-screen quad vertices
fn get_quad_vertex(index: u32) -> vec3<f32> {
    // Full-screen triangle strip covering NDC space
    let x = f32((index & 1u) << 2u) - 1.0;
    let y = f32((index & 2u) << 1u) - 1.0;
    return vec3<f32>(x, y, 0.0);
}

// Unproject point from NDC to world space
fn unproject_point(ndc: vec3<f32>) -> vec3<f32> {
    let world_pos = uniforms.inv_view_proj * vec4<f32>(ndc, 1.0);
    return world_pos.xyz / world_pos.w;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate full-screen quad vertex
    let pos = get_quad_vertex(input.vertex_index);
    out.clip_position = vec4<f32>(pos.xy, 0.0, 1.0);
    
    // Unproject near and far plane points for ray marching
    out.near_point = unproject_point(vec3<f32>(pos.xy, -1.0)); // Near plane (z = -1 in NDC)
    out.far_point = unproject_point(vec3<f32>(pos.xy, 1.0));   // Far plane (z = 1 in NDC)
    
    return out;
}

// Ray-plane intersection
fn ray_plane_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, plane_y: f32) -> f32 {
    let t = (plane_y - ray_origin.y) / ray_dir.y;
    return t;
}

// Grid pattern using fract and step
fn grid_pattern(coord: vec2<f32>, scale: f32) -> f32 {
    let grid = abs(fract(coord * scale - 0.5) - 0.5) / fwidth(coord * scale);
    let line_width = min(grid.x, grid.y);
    return 1.0 - min(line_width, 1.0);
}

// Axis lines (X: red, Z: blue)
fn axis_pattern(world_pos: vec3<f32>) -> vec4<f32> {
    let axis_width = 0.05;
    
    // X-axis (red)
    if (abs(world_pos.z) < axis_width) {
        return vec4<f32>(0.8, 0.1, 0.1, 1.0);
    }
    
    // Z-axis (blue)
    if (abs(world_pos.x) < axis_width) {
        return vec4<f32>(0.1, 0.4, 0.8, 1.0);
    }
    
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Ray from camera through pixel
    let ray_dir = normalize(input.far_point - input.near_point);
    
    // Grid plane at y = 0
    let plane_y = 0.0;
    let t = ray_plane_intersect(input.near_point, ray_dir, plane_y);
    
    // Discard if ray doesn't hit plane or hits behind camera
    if (t < 0.0) {
        discard;
    }
    
    // World space position on grid plane
    let world_pos = input.near_point + ray_dir * t;
    
    // Distance-based fade
    let distance = length(world_pos - uniforms.camera_pos);
    let fade_start = 50.0;
    let fade_end = 100.0;
    let fade = 1.0 - smoothstep(fade_start, fade_end, distance);
    
    if (fade < 0.01) {
        discard;
    }
    
    // Check for axis lines first
    let axis_color = axis_pattern(world_pos);
    if (axis_color.a > 0.0) {
        return vec4<f32>(axis_color.rgb, axis_color.a * fade);
    }
    
    // Major grid (every 10 units)
    let major_grid = grid_pattern(world_pos.xz, 0.1);
    
    // Minor grid (every 1 unit)
    let minor_grid = grid_pattern(world_pos.xz, 1.0);
    
    // Combine grids with different intensities
    let major_color = vec3<f32>(0.3, 0.3, 0.3);
    let minor_color = vec3<f32>(0.15, 0.15, 0.15);
    
    let grid_color = mix(minor_color * minor_grid, major_color, major_grid);
    let grid_alpha = max(major_grid * 0.8, minor_grid * 0.3);
    
    // Apply distance fade
    let final_alpha = grid_alpha * fade;
    
    if (final_alpha < 0.01) {
        discard;
    }
    
    return vec4<f32>(grid_color, final_alpha);
}

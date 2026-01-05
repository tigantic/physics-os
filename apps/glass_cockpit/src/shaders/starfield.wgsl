// Phase 7: Procedural Starfield Shader
// Renders a procedural starfield behind the globe
// Uses hash-based star placement with varying brightness

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct StarfieldUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: StarfieldUniforms;

// Generate full-screen quad vertices (covers entire screen)
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Full-screen triangle (3 vertices cover screen)
    let x = f32((input.vertex_index & 1u) << 2u) - 1.0;
    let y = f32((input.vertex_index & 2u) << 1u) - 1.0;
    
    out.clip_position = vec4<f32>(x, y, 0.9999, 1.0);  // Far plane
    out.uv = vec2<f32>(x, y);
    
    return out;
}

// High-quality hash functions for star placement
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract(vec2<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Procedural star layer - returns star brightness at this direction
fn star_layer(dir: vec3<f32>, density: f32, seed: f32) -> f32 {
    // Project direction to 2D with spherical coords
    let theta = atan2(dir.z, dir.x);  // Azimuth
    let phi = asin(clamp(dir.y, -1.0, 1.0));  // Elevation
    
    // Grid-based star placement (different grid sizes for variety)
    let grid_scale = density;
    let grid = vec2<f32>(theta, phi) * grid_scale + seed;
    let grid_id = floor(grid);
    let grid_fract = fract(grid);
    
    var brightness = 0.0;
    
    // Check this cell and neighbors
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cell = grid_id + vec2<f32>(f32(dx), f32(dy));
            
            // Random star position within cell
            let star_pos = hash22(cell + seed);
            let cell_offset = vec2<f32>(f32(dx), f32(dy)) + star_pos - grid_fract;
            
            // Star size varies
            let star_size = 0.015 + hash21(cell * 17.31 + seed) * 0.02;
            let dist = length(cell_offset);
            
            // Only ~30% of cells have stars
            let has_star = hash21(cell * 7.77 + seed) > 0.7;
            
            if (has_star && dist < star_size) {
                // Star brightness with variation
                let base_bright = hash21(cell * 13.37 + seed);
                // Core brightness falloff
                let falloff = 1.0 - smoothstep(0.0, star_size, dist);
                brightness = max(brightness, falloff * (0.3 + base_bright * 0.7));
            }
        }
    }
    
    return brightness;
}

// Star color based on temperature (blue-white-yellow-red)
fn star_color(seed: f32) -> vec3<f32> {
    let temp = hash21(vec2<f32>(seed, seed * 2.71));
    
    // Most stars white-ish, some blue, some warm
    if (temp < 0.1) {
        // Blue-white hot stars
        return vec3<f32>(0.7, 0.85, 1.0);
    } else if (temp > 0.9) {
        // Orange/red cool stars
        return vec3<f32>(1.0, 0.8, 0.6);
    } else {
        // White stars (most common)
        return vec3<f32>(1.0, 1.0, 0.98);
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Unproject screen UV to world ray direction
    let ndc = vec4<f32>(input.uv.x, input.uv.y, 1.0, 1.0);
    let world_pos = uniforms.inv_view_proj * ndc;
    var ray_dir = normalize(world_pos.xyz / world_pos.w - uniforms.camera_pos);
    
    // ═══ DAMPEN STARFIELD ROTATION ═══
    // Stars are very far away - they should barely move when orbiting
    // Mix with a "fixed" up direction to reduce apparent rotation
    let fixed_up = vec3<f32>(0.0, 1.0, 0.0);
    let dampen = 0.0003;  // Only 0.03% of actual camera rotation affects stars
    ray_dir = normalize(mix(vec3<f32>(ray_dir.x, 0.0, ray_dir.z), ray_dir, dampen) + fixed_up * ray_dir.y);
    
    // ═══ PARALLAX ON ZOOM ═══
    // Slight shift based on camera distance (zoom) - closer = more shift
    let cam_dist = length(uniforms.camera_pos);
    let parallax_offset = uniforms.camera_pos * 0.02 / max(cam_dist, 1.0);
    ray_dir = normalize(ray_dir + parallax_offset);
    
    // Multiple star layers at different densities for depth
    var total_brightness = 0.0;
    
    // Dense small stars (background)
    total_brightness += star_layer(ray_dir, 80.0, 0.0) * 0.3;
    
    // Medium stars
    total_brightness += star_layer(ray_dir, 40.0, 100.0) * 0.5;
    
    // Sparse bright stars (foreground)
    total_brightness += star_layer(ray_dir, 20.0, 200.0) * 0.8;
    
    // Very sparse super-bright stars
    total_brightness += star_layer(ray_dir, 10.0, 300.0) * 1.0;
    
    // Add subtle twinkling based on time (optional, subtle)
    let twinkle = 1.0 + sin(uniforms.time * 3.0 + hash31(ray_dir * 100.0) * 6.28) * 0.1;
    total_brightness *= twinkle;
    
    // Star color (mostly white with some variety)
    let color_seed = hash31(ray_dir * 50.0);
    let star_col = star_color(color_seed);
    
    // Final color with HDR-like bloom effect
    let final_color = star_col * total_brightness;
    
    // Add very subtle blue nebula tint in some areas
    let nebula = hash31(floor(ray_dir * 5.0)) * 0.02;
    let nebula_color = vec3<f32>(0.02, 0.02, 0.05) * nebula;
    
    return vec4<f32>(final_color + nebula_color, 1.0);
}

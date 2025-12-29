// Phase 2: Tensor Field Visualization Shader
// Renders colored tensor field overlay on base grid
// Constitutional compliance: Doctrine 1 (procedural), Doctrine 3 (GPU compute)

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    viewport: vec4<f32>,
};

struct TensorData {
    // Each tensor packed as 2x vec4
    // data[2*i]:   (xx, xy, xz, yy)
    // data[2*i+1]: (yz, zz, magnitude, trace)
    components: array<vec4<f32>>,
};

struct VisualizationParams {
    dimensions: vec3<u32>,      // Grid dimensions (width, height, depth)
    color_mode: u32,            // 0=Magnitude, 1=Trace, 2=Direction, 3=Heatmap
    intensity_scale: f32,
    threshold: f32,
    show_glyphs: u32,
    show_vectors: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> tensor_data: TensorData;
@group(1) @binding(1) var<uniform> vis_params: VisualizationParams;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) tensor_magnitude: f32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

// Get tensor at 3D grid index
fn get_tensor_index(x: u32, y: u32, z: u32) -> u32 {
    let w = vis_params.dimensions.x;
    let h = vis_params.dimensions.y;
    return z * h * w + y * w + x;
}

// Load tensor components from storage buffer
fn load_tensor(idx: u32) -> array<f32, 8> {
    let base = idx * 2u;
    let v0 = tensor_data.components[base];
    let v1 = tensor_data.components[base + 1u];
    
    var t: array<f32, 8>;
    t[0] = v0.x; // xx
    t[1] = v0.y; // xy
    t[2] = v0.z; // xz
    t[3] = v0.w; // yy
    t[4] = v1.x; // yz
    t[5] = v1.y; // zz
    t[6] = v1.z; // magnitude
    t[7] = v1.w; // trace
    return t;
}

// Map tensor to color based on visualization mode
fn tensor_to_color(tensor: array<f32, 8>, world_pos: vec3<f32>) -> vec4<f32> {
    let magnitude = tensor[6];
    let trace = tensor[7];
    
    // Apply threshold
    if magnitude < vis_params.threshold {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    var color = vec3<f32>(0.0);
    let scaled_magnitude = magnitude * vis_params.intensity_scale;
    
    // Color mode selection
    if vis_params.color_mode == 0u {
        // Magnitude mode: Blue (low) → Cyan → Green → Yellow → Red (high)
        let t = clamp(scaled_magnitude, 0.0, 1.0);
        if t < 0.25 {
            let s = t / 0.25;
            color = mix(vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.0, 0.5, 1.0), s);
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            color = mix(vec3<f32>(0.0, 0.5, 1.0), vec3<f32>(0.0, 1.0, 0.5), s);
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            color = mix(vec3<f32>(0.0, 1.0, 0.5), vec3<f32>(1.0, 1.0, 0.0), s);
        } else {
            let s = (t - 0.75) / 0.25;
            color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), s);
        }
    } else if vis_params.color_mode == 1u {
        // Trace mode: Diverging color map (red-white-blue)
        let t = clamp(trace * vis_params.intensity_scale * 0.5 + 0.5, 0.0, 1.0);
        if t < 0.5 {
            let s = t / 0.5;
            color = mix(vec3<f32>(0.8, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), s);
        } else {
            let s = (t - 0.5) / 0.5;
            color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 0.8), s);
        }
    } else if vis_params.color_mode == 2u {
        // Direction mode: Compute dominant eigenvector and map to RGB
        // Simplified: use (xx, yy, zz) as proxy for principal directions
        let dir = normalize(vec3<f32>(abs(tensor[0]), abs(tensor[3]), abs(tensor[5])));
        color = dir * scaled_magnitude;
    } else {
        // Heatmap mode: Orange-Yellow-White
        let t = clamp(scaled_magnitude, 0.0, 1.0);
        color = mix(vec3<f32>(1.0, 0.3, 0.0), vec3<f32>(1.0, 1.0, 1.0), t);
    }
    
    // Alpha based on magnitude
    let alpha = clamp(scaled_magnitude * 0.6, 0.1, 0.8);
    
    return vec4<f32>(color, alpha);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Instance index maps to 3D grid cell
    let total_cells = vis_params.dimensions.x * vis_params.dimensions.y * vis_params.dimensions.z;
    let idx = input.instance_idx;
    
    if idx >= total_cells {
        // Out of bounds - degenerate triangle
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.color = vec4<f32>(0.0);
        output.tensor_magnitude = 0.0;
        return output;
    }
    
    // Convert instance index to 3D grid coordinates
    let w = vis_params.dimensions.x;
    let h = vis_params.dimensions.y;
    let z = idx / (w * h);
    let rem = idx % (w * h);
    let y = rem / w;
    let x = rem % w;
    
    // Load tensor data
    let tensor = load_tensor(idx);
    
    // Compute cell center in world space
    // Map grid to [-10, 10] × [-10, 10] × [0, 10] region
    let grid_size = vec3<f32>(20.0, 20.0, 10.0);
    let grid_origin = vec3<f32>(-10.0, -10.0, 0.0);
    
    let fx = f32(x) / f32(vis_params.dimensions.x - 1u);
    let fy = f32(y) / f32(vis_params.dimensions.y - 1u);
    let fz = f32(z) / f32(max(vis_params.dimensions.z - 1u, 1u));
    
    let cell_center = grid_origin + vec3<f32>(fx, fy, fz) * grid_size;
    
    // Generate quad vertices (billboarded to camera)
    // Use switch instead of array indexing for compatibility
    var quad_uv: vec2<f32>;
    let vid = input.vertex_idx % 6u;
    switch vid {
        case 0u: { quad_uv = vec2<f32>(-0.5, -0.5); }
        case 1u: { quad_uv = vec2<f32>( 0.5, -0.5); }
        case 2u: { quad_uv = vec2<f32>( 0.5,  0.5); }
        case 3u: { quad_uv = vec2<f32>(-0.5, -0.5); }
        case 4u: { quad_uv = vec2<f32>( 0.5,  0.5); }
        case 5u: { quad_uv = vec2<f32>(-0.5,  0.5); }
        default: { quad_uv = vec2<f32>(0.0, 0.0); }
    }
    
    // Billboard: rotate quad to face camera
    let to_camera = normalize(camera.eye_pos.xyz - cell_center);
    let right = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), to_camera));
    let up = cross(to_camera, right);
    
    // Size based on tensor magnitude
    let magnitude = tensor[6];
    let size = magnitude * vis_params.intensity_scale * 0.4;
    
    let world_offset = right * quad_uv.x * size + up * quad_uv.y * size;
    let world_pos = cell_center + world_offset;
    
    output.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    output.world_pos = world_pos;
    output.color = tensor_to_color(tensor, cell_center);
    output.tensor_magnitude = magnitude;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    
    // Discard low-magnitude fragments
    if input.tensor_magnitude < vis_params.threshold {
        discard;
    }
    
    // Apply color with pre-multiplied alpha
    output.color = input.color;
    output.color.r *= output.color.a;
    output.color.g *= output.color.a;
    output.color.b *= output.color.a;
    
    return output;
}

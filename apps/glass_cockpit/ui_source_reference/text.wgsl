// Phase 2: Text Rendering Shader
// Instanced glyph rendering from bitmap atlas
// Constitutional compliance: Doctrine 1 (GPU instancing), Doctrine 8 (atlas texture)

@group(0) @binding(0)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(1)
var atlas_sampler: sampler;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec4<f32>,  // (x, y, width, height)
    @location(1) uv: vec4<f32>,        // (u, v, u_size, v_size)
    @location(2) color: vec4<f32>,     // (r, g, b, a)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

// Generate quad vertices procedurally (6 vertices per instance)
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Vertex positions for a quad (two triangles)
    var quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),  // Top-left
        vec2<f32>(1.0, 0.0),  // Top-right
        vec2<f32>(0.0, 1.0),  // Bottom-left
        vec2<f32>(0.0, 1.0),  // Bottom-left
        vec2<f32>(1.0, 0.0),  // Top-right
        vec2<f32>(1.0, 1.0),  // Bottom-right
    );
    
    let quad_pos = quad_positions[input.vertex_index];
    
    // Calculate screen position
    let screen_pos = vec2<f32>(
        input.position.x + quad_pos.x * input.position.z,
        input.position.y + quad_pos.y * input.position.w
    );
    
    // Convert to clip space (assuming 1920x1080 for now, will be passed as uniform in Phase 3)
    let clip_x = (screen_pos.x / 1920.0) * 2.0 - 1.0;
    let clip_y = 1.0 - (screen_pos.y / 1080.0) * 2.0;
    
    output.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    
    // Calculate texture coordinates
    output.tex_coords = vec2<f32>(
        input.uv.x + quad_pos.x * input.uv.z,
        input.uv.y + quad_pos.y * input.uv.w
    );
    
    output.color = input.color;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample atlas texture (R8 format, so alpha comes from red channel)
    let alpha = textureSample(atlas_texture, atlas_sampler, input.tex_coords).r;
    
    // Apply color with alpha from texture
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}

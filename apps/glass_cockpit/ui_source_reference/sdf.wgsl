// Phase 1: SDF (Signed Distance Field) UI Primitives
// Procedural shape rendering for Glass Cockpit rails and overlays
// Constitutional compliance: Doctrine 1 (procedural), Doctrine 8 (minimal memory)

// Uniforms for UI rendering
struct UiUniforms {
    screen_size: vec2<f32>,
    time: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> ui: UiUniforms;

// Vertex shader output / Fragment shader input
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen quad vertex shader (procedural - no vertex buffers)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate full-screen quad positions
    let x = f32((vertex_index & 1u) << 1u) - 1.0;
    let y = f32((vertex_index & 2u)) - 1.0;
    
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    
    return output;
}

// ============================================================================
// SDF PRIMITIVE FUNCTIONS
// ============================================================================

// Signed distance to a rectangle with rounded corners
// Returns negative inside, positive outside
fn sdf_rounded_rect(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let d = abs(p) - size + vec2<f32>(radius);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

// Signed distance to a circle
fn sdf_circle(p: vec2<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

// Signed distance to a line segment
fn sdf_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, thickness: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - thickness;
}

// ============================================================================
// SDF OPERATIONS
// ============================================================================

// Smooth minimum (union)
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * k * (1.0 / 6.0);
}

// Smooth maximum (intersection)
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_min(-a, -b, k);
}

// Smooth subtraction
fn smooth_subtract(a: f32, b: f32, k: f32) -> f32 {
    return smooth_max(-a, b, k);
}

// ============================================================================
// ANTIALIASING AND RENDERING
// ============================================================================

// Convert SDF distance to alpha with antialiasing
fn sdf_to_alpha(distance: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(-width, width, distance);
}

// Derivative-based antialiasing (better for detailed shapes)
fn sdf_to_alpha_aa(distance: f32) -> f32 {
    let aa_width = fwidth(distance);
    return 1.0 - smoothstep(-aa_width, aa_width, distance);
}

// ============================================================================
// UI RENDERING FUNCTIONS
// ============================================================================

// Render left rail container (10% of screen width)
fn render_left_rail(pixel_pos: vec2<f32>) -> vec4<f32> {
    let rail_width = ui.screen_size.x * 0.10;
    
    // Position relative to rail (origin at top-left of rail)
    let rail_pos = pixel_pos - vec2<f32>(0.0, 0.0);
    
    // Centered within rail
    let center = vec2<f32>(rail_width * 0.5, ui.screen_size.y * 0.5);
    let p = rail_pos - center;
    
    // Rail container: rounded rectangle
    let size = vec2<f32>(rail_width * 0.9, ui.screen_size.y * 0.95);
    let corner_radius = 8.0;
    let dist = sdf_rounded_rect(p, size * 0.5, corner_radius);
    
    // Render with antialiasing
    let alpha = sdf_to_alpha_aa(dist);
    
    // Dark semi-transparent background
    let bg_color = vec3<f32>(0.1, 0.1, 0.12);
    return vec4<f32>(bg_color, alpha * 0.85);
}

// Render right rail container (10% of screen width)
fn render_right_rail(pixel_pos: vec2<f32>) -> vec4<f32> {
    let rail_width = ui.screen_size.x * 0.10;
    let rail_x_start = ui.screen_size.x - rail_width;
    
    // Position relative to rail
    let rail_pos = pixel_pos - vec2<f32>(rail_x_start, 0.0);
    
    // Centered within rail
    let center = vec2<f32>(rail_width * 0.5, ui.screen_size.y * 0.5);
    let p = rail_pos - center;
    
    // Rail container: rounded rectangle
    let size = vec2<f32>(rail_width * 0.9, ui.screen_size.y * 0.95);
    let corner_radius = 8.0;
    let dist = sdf_rounded_rect(p, size * 0.5, corner_radius);
    
    // Render with antialiasing
    let alpha = sdf_to_alpha_aa(dist);
    
    // Dark semi-transparent background (slightly different shade for variety)
    let bg_color = vec3<f32>(0.12, 0.1, 0.1);
    return vec4<f32>(bg_color, alpha * 0.85);
}

// Render telemetry card within a rail
fn render_telemetry_card(p: vec2<f32>, center: vec2<f32>, width: f32, height: f32) -> vec4<f32> {
    let pos = p - center;
    let size = vec2<f32>(width, height);
    let corner_radius = 6.0;
    let dist = sdf_rounded_rect(pos, size * 0.5, corner_radius);
    
    let alpha = sdf_to_alpha_aa(dist);
    
    // Slightly lighter than rail background
    let card_color = vec3<f32>(0.15, 0.15, 0.17);
    return vec4<f32>(card_color, alpha * 0.95);
}

// ============================================================================
// MAIN FRAGMENT SHADER
// ============================================================================

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Pixel position in screen space
    let pixel_pos = input.uv * ui.screen_size;
    
    // Calculate rail boundaries
    let left_rail_width = ui.screen_size.x * 0.10;
    let right_rail_start = ui.screen_size.x * 0.90;
    
    var final_color = vec4<f32>(0.0);
    
    // Render left rail
    if pixel_pos.x < left_rail_width {
        final_color = render_left_rail(pixel_pos);
        
        // Add telemetry cards in left rail
        let rail_center_x = left_rail_width * 0.5;
        let card_width = left_rail_width * 0.85;
        
        // P-core card (top)
        let p_core_card = render_telemetry_card(
            pixel_pos,
            vec2<f32>(rail_center_x, 80.0),
            card_width,
            60.0
        );
        final_color = mix(final_color, p_core_card, p_core_card.a);
        
        // E-core card
        let e_core_card = render_telemetry_card(
            pixel_pos,
            vec2<f32>(rail_center_x, 160.0),
            card_width,
            60.0
        );
        final_color = mix(final_color, e_core_card, e_core_card.a);
        
        // FPS card
        let fps_card = render_telemetry_card(
            pixel_pos,
            vec2<f32>(rail_center_x, 240.0),
            card_width,
            60.0
        );
        final_color = mix(final_color, fps_card, fps_card.a);
    }
    // Render right rail
    else if pixel_pos.x > right_rail_start {
        final_color = render_right_rail(pixel_pos);
        
        // Add telemetry cards in right rail
        let rail_center_x = right_rail_start + (ui.screen_size.x - right_rail_start) * 0.5;
        let card_width = (ui.screen_size.x - right_rail_start) * 0.85;
        
        // Memory card
        let memory_card = render_telemetry_card(
            pixel_pos,
            vec2<f32>(rail_center_x, 80.0),
            card_width,
            60.0
        );
        final_color = mix(final_color, memory_card, memory_card.a);
        
        // Stability card
        let stability_card = render_telemetry_card(
            pixel_pos,
            vec2<f32>(rail_center_x, 160.0),
            card_width,
            60.0
        );
        final_color = mix(final_color, stability_card, stability_card.a);
    }
    
    return final_color;
}

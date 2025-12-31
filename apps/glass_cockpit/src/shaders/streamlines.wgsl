// Phase 5: Streamline Shader
// Phase 8: Updated for globe projection (geodetic → ECEF → clip space)
// Renders precomputed streamlines as ribbons with magnitude-based thickness
// Constitutional compliance: Doctrine 3 (GPU rendering)

// ============================================================================
// TYPES
// ============================================================================

struct StreamlineUniforms {
    bounds: vec4<f32>,     // (lon_min, lon_max, lat_min, lat_max)
    stats: vec4<f32>,      // (max_speed, max_vorticity, base_thickness, speed_factor)
    time: vec4<f32>,       // (current_time, animation_speed, view_proj_00, view_proj_01)
}

// Camera uniforms - shared with globe pipeline via bind group 1
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
    camera_pos_high: vec3<f32>,
    _padding1b: f32,
    camera_pos_low: vec3<f32>,
    _padding1c: f32,
    zoom: f32,
    aspect_ratio: f32,
    time: f32,
    _padding2: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,    // (lon, lat, altitude) in DEGREES
    @location(1) tangent: vec3<f32>,     // normalized tangent (in lon/lat space)
    @location(2) properties: vec4<f32>,  // (speed, vorticity, arc_length, side)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) arc_length: f32,
    @location(2) edge_dist: f32,  // Distance from edge for anti-aliasing
    @location(3) world_pos: vec3<f32>,  // For backface culling
}

// ============================================================================
// BINDINGS
// ============================================================================

@group(0) @binding(0) var<uniform> uniforms: StreamlineUniforms;
@group(1) @binding(0) var<uniform> camera: CameraUniforms;

// ============================================================================
// GLOBE PROJECTION
// ============================================================================

const PI: f32 = 3.14159265359;
const GLOBE_RADIUS: f32 = 1.0;  // Must match GlobeConfig::default().radius

// Convert geodetic (lon, lat in radians) to ECEF (x, y, z)
// Y-up convention to match globe_quadtree.rs mesh generation
fn geodetic_to_ecef(lon: f32, lat: f32, radius: f32) -> vec3<f32> {
    // Y-up: Y = sin(lat), XZ plane is equator
    return vec3<f32>(
        radius * cos(lat) * cos(lon),  // X: toward 0°N 0°E
        radius * sin(lat),              // Y: toward North Pole (Y-up)
        radius * cos(lat) * sin(lon)    // Z: toward 0°N 90°E
    );
}

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
    
    // Convert input position (lon, lat in degrees) to radians
    let lon_rad = in.position.x * PI / 180.0;
    let lat_rad = in.position.y * PI / 180.0;
    
    // Convert geodetic to 3D position on globe surface
    // Slight offset above surface to prevent z-fighting
    let world_pos = geodetic_to_ecef(lon_rad, lat_rad, GLOBE_RADIUS * 1.002);
    
    // Calculate ribbon width based on speed
    let base_thickness = stats.z;
    let speed_factor = stats.w;
    let max_speed = stats.x;
    let speed_norm = clamp(in.properties.x / max_speed, 0.0, 1.0);
    let thickness = base_thickness * (1.0 + speed_factor * speed_norm) * 0.003; // World units
    
    // Calculate perpendicular direction in world space for ribbon expansion
    // Tangent is in lon/lat space - convert to world space
    let tangent_lon = in.tangent.x * PI / 180.0;
    let tangent_lat = in.tangent.y * PI / 180.0;
    
    // Approximate tangent in world space using partial derivatives
    let east = normalize(vec3<f32>(-sin(lon_rad), 0.0, cos(lon_rad)));
    let north = normalize(vec3<f32>(-cos(lon_rad) * sin(lat_rad), cos(lat_rad), -sin(lon_rad) * sin(lat_rad)));
    let world_tangent = normalize(east * tangent_lon + north * tangent_lat);
    
    // Perpendicular to tangent, on the sphere surface (cross with normal)
    let surface_normal = normalize(world_pos);
    let perp = normalize(cross(surface_normal, world_tangent));
    
    // Offset for ribbon side
    let ribbon_offset = perp * thickness * in.properties.w; // side: -1 or 1
    let offset_pos = world_pos + ribbon_offset;
    
    // Backface culling - hide streamlines on far side of globe
    let to_camera = normalize(camera.camera_pos - world_pos);
    let facing = dot(surface_normal, to_camera);
    
    // Phase 8: Apply RTE (Relative-To-Eye) transformation to match globe shader
    // This ensures streamlines render at the same coordinates as the globe surface
    let rte_pos = offset_pos - camera.camera_pos_high + (-camera.camera_pos_low);
    
    // Project to clip space using camera view_proj matrix
    let clip_pos = camera.view_proj * vec4<f32>(rte_pos, 1.0);
    
    // Animated dash pattern
    let time = uniforms.time.x;
    let anim_speed = uniforms.time.y;
    let arc_offset = in.properties.z - time * anim_speed * 0.0001;
    
    var out: VertexOutput;
    out.clip_position = clip_pos;
    out.world_pos = world_pos;
    
    // Color based on speed with vorticity overlay
    let base_color = speed_to_color(in.properties.x, max_speed);
    let color = vorticity_overlay(base_color, in.properties.y, stats.y);
    
    // Apply backface fade (hide far side of globe)
    let alpha = select(0.0, 0.85, facing > 0.0);
    out.color = vec4<f32>(color, alpha);
    
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
    
    // Backface culling alpha was set in vertex shader
    // If alpha is 0, we're on the far side of the globe - discard
    if color.a < 0.01 {
        discard;
    }
    
    // Anti-aliased edge (for ribbon width)
    let edge_fade = 1.0 - smoothstep(0.7, 1.0, abs(in.edge_dist));
    color.a *= edge_fade;
    
    // Subtle center highlight
    let center_highlight = 1.0 - abs(in.edge_dist);
    let highlight = vec3<f32>(0.1) * center_highlight * center_highlight;
    color = vec4<f32>(color.r + highlight.x, color.g + highlight.y, color.b + highlight.z, color.a);
    
    if color.a < 0.01 {
        discard;
    }
    
    return color;
}

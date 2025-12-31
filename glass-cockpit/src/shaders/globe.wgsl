// Phase 4: Globe Projection Shader
// Phase 8: Appendix F - RTE coordinate precision pipeline
// ECEF coordinate conversion with camera-relative positioning
// Constitutional compliance: Doctrine 3 (GPU compute), Doctrine 1 (no CPU dependency)

// Vertex shader uniforms
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    camera_pos_ecef: vec3<f32>,  // Camera position in ECEF meters (f32 portion)
    _padding: f32,
    // Phase 8: Split precision for RTE - high bits
    camera_pos_high: vec3<f32>,  // High-order bits of camera position
    _padding1b: f32,
    // Phase 8: Split precision for RTE - low bits
    camera_pos_low: vec3<f32>,   // Low-order bits (remainder after f32 cast)
    _padding1c: f32,
    zoom: f32,
    aspect_ratio: f32,
    time: f32,
    _padding2: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Vertex input (globe mesh)
// Phase 3: Added tile_layer for texture array streaming
struct VertexInput {
    @location(0) position: vec3<f32>,  // ECEF position in meters
    @location(1) normal: vec3<f32>,    // Normalized normal vector
    @location(2) uv: vec2<f32>,        // Texture coordinates (0-1)
    @location(3) lat_lon: vec2<f32>,   // Latitude, Longitude in radians
    @location(4) tile_layer: f32,      // Texture array layer (-1 = procedural)
    @location(5) _padding: vec3<f32>,  // Alignment padding
}

// Vertex output
// Phase 3: Pass tile_layer to fragment shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) lat_lon: vec2<f32>,
    @location(4) tile_layer: f32,      // Texture array layer
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 8: Appendix F - Relative-to-Eye (RTE) Transformation
// Double-single arithmetic: maintains f64 precision using two f32 values
// See Sovereign_UI.md F.4.2 for specification
// ═══════════════════════════════════════════════════════════════════════

// Vertex shader: Transform globe vertices with RTE coordinates
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Phase 8: Relative-To-Eye (RTE) transformation with split precision
    // Formula: rel = (world_pos - camera_high) + (-camera_low)
    // This maintains sub-meter precision at planetary scale
    let rel_high = input.position - camera.camera_pos_high;
    let rel_low = -camera.camera_pos_low;  // World pos has no low bits (gridded data)
    let rte_position = rel_high + rel_low;
    
    // Transform to clip space
    output.position = camera.view_proj * vec4<f32>(rte_position, 1.0);
    
    // Pass through world position for fragment shader
    output.world_pos = rte_position;
    output.normal = input.normal;
    output.uv = input.uv;
    output.lat_lon = input.lat_lon;
    // Phase 3: Pass texture layer for array sampling
    output.tile_layer = input.tile_layer;
    
    return output;
}

// Fragment shader uniforms
struct MaterialUniforms {
    base_color: vec4<f32>,
    water_color: vec4<f32>,
    grid_color: vec4<f32>,
    grid_thickness: f32,
    latitude_spacing: f32,  // Degrees between latitude lines
    longitude_spacing: f32, // Degrees between longitude lines
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> material: MaterialUniforms;

// Phase 3: Texture array for streaming satellite tiles
// 256x256 × 128 layers - see tile_texture_array.rs
@group(1) @binding(1)
var satellite_textures: texture_2d_array<f32>;

@group(1) @binding(2)
var satellite_sampler: sampler;

// Phase 3: Fallback single texture for legacy code paths
@group(1) @binding(3)
var satellite_texture_single: texture_2d<f32>;

// Fragment shader: SATELLITE MODE with Real Tile Streaming
// Samples actual NASA GIBS tiles with tech-grid overlay
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // ═══════════════════════════════════════════════════════════════
    // 1. LIGHTING SETUP
    // ═══════════════════════════════════════════════════════════════
    let light_dir = normalize(vec3<f32>(1.0, 0.5, 0.5));
    let view_dir = normalize(camera.camera_pos_ecef - input.world_pos);
    let normal = normalize(input.normal);
    let n_dot_l = dot(normal, light_dir);
    let day_night = smoothstep(-0.1, 0.1, n_dot_l); 

    // ═══════════════════════════════════════════════════════════════
    // 2. TECH GRID CALCULATION (10° spacing)
    // ═══════════════════════════════════════════════════════════════
    let lat = degrees(input.lat_lon.x);
    let lon = degrees(input.lat_lon.y);
    let lat_grid = abs(fract(lat / 10.0) - 0.5) * 20.0;
    let lon_grid = abs(fract(lon / 10.0) - 0.5) * 20.0;
    let grid_color = vec3<f32>(0.0, 0.8, 1.0) * 0.3; 
    let is_grid = min(lat_grid, lon_grid) < 0.05;
    let grid_overlay = select(vec3<f32>(0.0), grid_color, is_grid);

    // ═══════════════════════════════════════════════════════════════
    // 3. TEXTURE SAMPLING (Real tiles or placeholder)
    // ═══════════════════════════════════════════════════════════════
    let layer_idx = i32(input.tile_layer + 0.5);
    var surface_color: vec3<f32>;
    if (layer_idx >= 0) {
        // Sample from streamed satellite tile
        let sat_sample = textureSample(satellite_textures, satellite_sampler, input.uv, layer_idx).rgb;
        // Full brightness satellite imagery with grid overlay
        surface_color = sat_sample + grid_overlay * 0.3; 
    } else {
        // Fallback: deep grey with bright grid lines
        let deep_grey = vec3<f32>(0.02, 0.02, 0.04);
        let fallback_grid = select(deep_grey, grid_color * 2.0, is_grid);
        surface_color = fallback_grid;
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. COMPOSITE DAY/NIGHT
    // ═══════════════════════════════════════════════════════════════
    let night_color = vec3<f32>(0.002, 0.002, 0.005);
    let lit_surface = mix(night_color, surface_color, day_night);

    // ═══════════════════════════════════════════════════════════════
    // 5. THIN BLUE ATMOSPHERE RIM
    // ═══════════════════════════════════════════════════════════════
    let fresnel = 1.0 - max(dot(view_dir, normal), 0.0);
    let rim_glow = pow(fresnel, 6.0) * vec3<f32>(0.0, 0.3, 0.8) * 2.0;

    return vec4<f32>(lit_surface + rim_glow, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════
// COMMAND CENTER FALLBACK - Dark Tech-Grid Aesthetic
// When satellite textures unavailable: tactical holographic globe
// ═══════════════════════════════════════════════════════════════════════
@fragment
fn fs_procedural(input: VertexOutput) -> @location(0) vec4<f32> {
    let lat = input.lat_lon.x;
    let lon = input.lat_lon.y;
    let lat_deg = degrees(lat);
    let lon_deg = degrees(lon);
    
    // ═══ DEEP SPACE BASE ═══
    // Near-black with subtle blue undertone (command center aesthetic)
    let base_color = vec3<f32>(0.03, 0.03, 0.05);
    
    // ═══ TECH-GRID OVERLAY ═══
    // 10° primary grid, 2° secondary grid
    let primary_spacing = 10.0;
    let secondary_spacing = 2.0;
    
    // Primary grid (bright cyan)
    let lat_primary = abs(fract(lat_deg / primary_spacing) - 0.5) * 2.0;
    let lon_primary = abs(fract(lon_deg / primary_spacing) - 0.5) * 2.0;
    let primary_line = smoothstep(0.03, 0.0, min(lat_primary, lon_primary));
    let primary_color = vec3<f32>(0.0, 0.6, 0.8) * primary_line * 0.4;
    
    // Secondary grid (dim cyan)
    let lat_secondary = abs(fract(lat_deg / secondary_spacing) - 0.5) * 2.0;
    let lon_secondary = abs(fract(lon_deg / secondary_spacing) - 0.5) * 2.0;
    let secondary_line = smoothstep(0.08, 0.0, min(lat_secondary, lon_secondary));
    let secondary_color = vec3<f32>(0.0, 0.3, 0.5) * secondary_line * 0.15;
    
    // ═══ EQUATOR + TROPICS HIGHLIGHT ═══
    let equator_glow = exp(-pow(lat_deg / 2.0, 2.0)) * 0.3;
    let tropic_cancer = exp(-pow((lat_deg - 23.5) / 1.5, 2.0)) * 0.15;
    let tropic_capricorn = exp(-pow((lat_deg + 23.5) / 1.5, 2.0)) * 0.15;
    let reference_lines = vec3<f32>(0.0, 0.4, 0.6) * (equator_glow + tropic_cancer + tropic_capricorn);
    
    // ═══ POLAR CIRCLES ═══
    let arctic = exp(-pow((lat_deg - 66.5) / 1.5, 2.0)) * 0.12;
    let antarctic = exp(-pow((lat_deg + 66.5) / 1.5, 2.0)) * 0.12;
    let polar_lines = vec3<f32>(0.3, 0.5, 0.7) * (arctic + antarctic);
    
    // ═══ SUBTLE HEX PATTERN (tactical overlay) ═══
    let hex_scale = 8.0;
    let hex_uv = vec2<f32>(lon_deg * hex_scale / 360.0, lat_deg * hex_scale / 180.0);
    let hex_noise = fbm_noise_2d(hex_uv, 2) * 0.5 + 0.5;
    let hex_pattern = smoothstep(0.48, 0.52, hex_noise) * 0.05;
    let hex_color = vec3<f32>(0.1, 0.2, 0.3) * hex_pattern;
    
    // ═══ COMBINE LAYERS ═══
    var surface = base_color + primary_color + secondary_color + reference_lines + polar_lines + hex_color;
    
    // ═══ RIM GLOW (Sharp holographic edge) ═══
    let view_dir = normalize(input.world_pos);
    let normal = normalize(input.normal);
    let fresnel = 1.0 - max(dot(-view_dir, normal), 0.0);
    let rim = pow(fresnel, 6.0) * 1.5;
    let rim_color = vec3<f32>(0.0, 0.4, 1.0) * rim;
    
    // ═══ MINIMAL LIGHTING (Command center ambiance) ═══
    // No harsh sun - just subtle hemispheric illumination
    let ambient = 0.7;
    let hemisphere = max(dot(normal, vec3<f32>(0.0, 1.0, 0.0)) * 0.15 + 0.85, 0.0);
    
    let final_color = surface * ambient * hemisphere + rim_color;
    
    return vec4<f32>(final_color, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════
// NOISE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════

fn hash_2d(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(hash_2d(i + vec2<f32>(0.0, 0.0)), hash_2d(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash_2d(i + vec2<f32>(0.0, 1.0)), hash_2d(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    ) * 2.0 - 1.0;
}

fn fbm_noise_2d(p: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise_2d(pos);
        pos *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

// Elliptical continent blob - CORRECTED for spherical geometry
// Fixes equirectangular distortion at high latitudes
fn continent_blob(lon: f32, lat: f32, center_lon: f32, center_lat: f32, width: f32, height: f32) -> f32 {
    // Handle longitude wrapping
    var d_lon = lon - center_lon;
    if (d_lon > 180.0) { d_lon -= 360.0; }
    if (d_lon < -180.0) { d_lon += 360.0; }
    
    let d_lat = lat - center_lat;
    
    // ═══ SPHERICAL CORRECTION ═══
    // Longitude degrees shrink toward poles: 1° lon = cos(lat) * 111km
    // Apply cosine correction at the CENTER latitude of the blob
    let cos_correction = cos(radians(center_lat));
    let effective_d_lon = d_lon * cos_correction;
    
    // Use corrected longitude distance
    let dist = sqrt(pow(effective_d_lon / width, 2.0) + pow(d_lat / height, 2.0));
    return smoothstep(1.2, 0.5, dist);
}

// Helper functions for coordinate conversions

// Convert ECEF to geodetic (lat/lon/height)
fn ecef_to_geodetic(ecef: vec3<f32>) -> vec3<f32> {
    // WGS84 ellipsoid parameters
    let a = 6378137.0;  // Equatorial radius (meters)
    let b = 6356752.314245;  // Polar radius (meters)
    let e2 = 1.0 - (b * b) / (a * a);  // First eccentricity squared
    
    let p = sqrt(ecef.x * ecef.x + ecef.z * ecef.z);
    let theta = atan2(ecef.y * a, p * b);
    
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);
    
    let lat = atan2(
        ecef.y + e2 * b * sin_theta * sin_theta * sin_theta,
        p - e2 * a * cos_theta * cos_theta * cos_theta
    );
    
    let lon = atan2(ecef.z, ecef.x);
    
    let sin_lat = sin(lat);
    let N = a / sqrt(1.0 - e2 * sin_lat * sin_lat);
    let height = p / cos(lat) - N;
    
    return vec3<f32>(lat, lon, height);
}

// Convert geodetic (lat/lon/height) to ECEF
fn geodetic_to_ecef(lat: f32, lon: f32, height: f32) -> vec3<f32> {
    // WGS84 ellipsoid parameters
    let a = 6378137.0;  // Equatorial radius (meters)
    let e2 = 0.00669437999014;  // First eccentricity squared
    
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);
    
    let N = a / sqrt(1.0 - e2 * sin_lat * sin_lat);
    
    let x = (N + height) * cos_lat * cos_lon;
    let y = (N + height) * cos_lat * sin_lon;
    let z = (N * (1.0 - e2) + height) * sin_lat;
    
    return vec3<f32>(x, y, z);
}

// Atmospheric scattering shader (optional enhancement)
// Phase 3: Updated to use texture array with fallback
@fragment
fn fs_atmosphere(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple atmospheric scattering effect
    let view_dir = normalize(input.world_pos);
    let normal = normalize(input.normal);
    
    // Fresnel effect for atmosphere
    let fresnel = pow(1.0 - max(dot(view_dir, normal), 0.0), 3.0);
    
    let atmosphere_color = vec3<f32>(0.5, 0.7, 1.0);
    
    // Phase 3: Sample from texture array if layer valid
    var base_color: vec3<f32>;
    if (input.tile_layer >= 0.0) {
        let layer = i32(input.tile_layer);
        base_color = textureSample(satellite_textures, satellite_sampler, input.uv, layer).rgb;
    } else {
        base_color = textureSample(satellite_texture_single, satellite_sampler, input.uv).rgb;
    }
    
    // Lighting
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let lighting = 0.3 + 0.7 * n_dot_l;
    
    // Combine base color with atmosphere
    let final_color = mix(
        base_color * lighting,
        atmosphere_color,
        fresnel * 0.3
    );
    
    return vec4<f32>(final_color, 1.0);
}

// Simple satellite texture shader (Sprint 2)
// Phase 3: Updated to use texture array
// Constants for UV computation
const PI_SAT: f32 = 3.14159265359;

@fragment
fn fs_satellite(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use the interpolated lat/lon directly from vertex data
    // These are fixed to the Earth's surface and don't move with camera
    let lat = input.lat_lon.x;
    let lon = input.lat_lon.y;
    
    // Simple equirectangular UV mapping
    // The vertex lat/lon are already correct for the Earth's surface
    let u = (lon + PI_SAT) / (2.0 * PI_SAT);
    let v = (lat + PI_SAT / 2.0) / PI_SAT;
    
    // Phase 3: Sample from texture array if layer valid
    var tex_color: vec4<f32>;
    if (input.tile_layer >= 0.0) {
        let layer = i32(input.tile_layer);
        tex_color = textureSample(satellite_textures, satellite_sampler, vec2<f32>(u, v), layer);
    } else {
        tex_color = textureSample(satellite_texture_single, satellite_sampler, vec2<f32>(u, v));
    }
    let light_dir = normalize(vec3<f32>(0.5, 0.3, 0.8));
    let normal = normalize(input.normal);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let ambient = 0.05;  // Almost black in shadow (space is dark!)
    let diffuse = 2.0 * n_dot_l;  // Bright sun on the day side
    let lighting = ambient + diffuse;
    
    // ═══════════════════════════════════════════════════════════════════════
    // CRUSH THE LEVELS (God's Eye / Dark Matter tint)
    // We want the earth to look almost unlit - vector data should POP
    // ═══════════════════════════════════════════════════════════════════════
    let dark_matter_tint = vec3<f32>(0.2, 0.2, 0.2);
    let base_color = tex_color.rgb * dark_matter_tint * lighting;
    
    // ═══════════════════════════════════════════════════════════════════════
    // TIGHTEN THE RIM (Sharp atmosphere edge, not foggy)
    // ═══════════════════════════════════════════════════════════════════════
    
    // View direction from camera to this fragment
    let view_dir = normalize(camera.camera_pos_high - input.world_pos);
    
    // Rim factor: how close is this pixel to the edge of the sphere?
    let rim_factor = 1.0 - max(dot(view_dir, normal), 0.0);
    
    // Atmosphere color (Deep Space Blue - darker than before)
    let atmosphere_color = vec3<f32>(0.0, 0.2, 0.6);
    
    // Power 8.0 = razor-thin edge glow, clears the center view
    let glow_intensity = pow(rim_factor, 8.0) * 2.0;
    
    // Composite: dark earth + thin edge glow
    let final_color = base_color + (atmosphere_color * glow_intensity);
    
    return vec4<f32>(final_color, 1.0);
}

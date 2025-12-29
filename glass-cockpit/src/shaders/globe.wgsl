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
struct VertexInput {
    @location(0) position: vec3<f32>,  // ECEF position in meters
    @location(1) normal: vec3<f32>,    // Normalized normal vector
    @location(2) uv: vec2<f32>,        // Texture coordinates (0-1)
    @location(3) lat_lon: vec2<f32>,   // Latitude, Longitude in radians
}

// Vertex output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) lat_lon: vec2<f32>,
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

@group(1) @binding(1)
var satellite_texture: texture_2d<f32>;

@group(1) @binding(2)
var satellite_sampler: sampler;

// Fragment shader: Render globe with satellite texture and grid
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample satellite texture
    let tex_color = textureSample(satellite_texture, satellite_sampler, input.uv);
    
    // Calculate lighting (simple directional light)
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let n_dot_l = max(dot(normalize(input.normal), light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7 * n_dot_l;
    let lighting = ambient + diffuse;
    
    // Base color with lighting
    var color = tex_color * vec4<f32>(vec3<f32>(lighting), 1.0);
    
    // Add latitude/longitude grid overlay
    let lat_deg = degrees(input.lat_lon.x);
    let lon_deg = degrees(input.lat_lon.y);
    
    let lat_mod = abs(fract(lat_deg / material.latitude_spacing) - 0.5) * 2.0;
    let lon_mod = abs(fract(lon_deg / material.longitude_spacing) - 0.5) * 2.0;
    
    let grid_threshold = material.grid_thickness;
    
    if (lat_mod < grid_threshold || lon_mod < grid_threshold) {
        // Grid line - blend with base color
        color = mix(color, material.grid_color, 0.5);
    }
    
    return color;
}

// Procedural Earth texture (fallback when satellite tiles unavailable)
@fragment
fn fs_procedural(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple procedural ocean/land pattern based on lat/lon
    let lat = input.lat_lon.x;
    let lon = input.lat_lon.y;
    
    // Perlin-like noise function (simplified)
    let noise_scale = 5.0;
    let noise_val = sin(lat * noise_scale) * cos(lon * noise_scale * 2.0);
    
    // Ocean vs land threshold
    let is_land = noise_val > 0.1;
    
    // Base colors
    let ocean_color = vec3<f32>(0.1, 0.2, 0.4);
    let land_color = vec3<f32>(0.3, 0.5, 0.2);
    
    var base_color: vec3<f32>;
    if (is_land) {
        base_color = land_color;
    } else {
        base_color = ocean_color;
    }
    
    // Lighting
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let n_dot_l = max(dot(normalize(input.normal), light_dir), 0.0);
    let lighting = 0.3 + 0.7 * n_dot_l;
    
    var color = vec4<f32>(base_color * lighting, 1.0);
    
    // Add grid (15 degree spacing, no material uniforms)
    let lat_deg = degrees(lat);
    let lon_deg = degrees(lon);
    
    let lat_spacing = 15.0;
    let lon_spacing = 15.0;
    let grid_thickness = 0.05;
    
    let lat_mod = abs(fract(lat_deg / lat_spacing) - 0.5) * 2.0;
    let lon_mod = abs(fract(lon_deg / lon_spacing) - 0.5) * 2.0;
    
    if (lat_mod < grid_thickness || lon_mod < grid_thickness) {
        let grid_color = vec4<f32>(0.5, 0.5, 0.5, 1.0);
        color = mix(color, grid_color, 0.5);
    }
    
    return color;
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
@fragment
fn fs_atmosphere(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple atmospheric scattering effect
    let view_dir = normalize(input.world_pos);
    let normal = normalize(input.normal);
    
    // Fresnel effect for atmosphere
    let fresnel = pow(1.0 - max(dot(view_dir, normal), 0.0), 3.0);
    
    let atmosphere_color = vec3<f32>(0.5, 0.7, 1.0);
    let base_color = textureSample(satellite_texture, satellite_sampler, input.uv).rgb;
    
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

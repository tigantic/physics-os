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

// Procedural Earth texture - photorealistic approximation
// Uses multi-octave noise + spherical harmonics for continent shapes
@fragment
fn fs_procedural(input: VertexOutput) -> @location(0) vec4<f32> {
    let lat = input.lat_lon.x;
    let lon = input.lat_lon.y;
    let lat_deg = degrees(lat);
    let lon_deg = degrees(lon);
    
    // ═══════════════════════════════════════════════════════════════
    // PROCEDURAL CONTINENT GENERATION
    // Approximates major landmasses using spherical harmonics
    // ═══════════════════════════════════════════════════════════════
    
    // Multi-octave noise for terrain variation
    var terrain = 0.0;
    terrain += 0.5 * fbm_noise_2d(vec2<f32>(lon * 2.0, lat * 2.0), 4);
    terrain += 0.3 * fbm_noise_2d(vec2<f32>(lon * 5.0, lat * 5.0), 3);
    terrain += 0.2 * fbm_noise_2d(vec2<f32>(lon * 12.0, lat * 12.0), 2);
    
    // Continent masks using spherical harmonics approximation
    // These create rough shapes matching real continents
    var land_mask = 0.0;
    
    // North America (centered around -100°, 45°)
    land_mask += continent_blob(lon_deg, lat_deg, -100.0, 45.0, 35.0, 25.0);
    land_mask += continent_blob(lon_deg, lat_deg, -105.0, 55.0, 25.0, 15.0); // Canada
    land_mask += continent_blob(lon_deg, lat_deg, -90.0, 30.0, 15.0, 10.0);  // Gulf states
    
    // South America (centered around -60°, -15°)
    land_mask += continent_blob(lon_deg, lat_deg, -60.0, -15.0, 20.0, 35.0);
    land_mask += continent_blob(lon_deg, lat_deg, -70.0, -35.0, 12.0, 20.0); // Chile/Argentina
    
    // Europe (centered around 15°, 50°)
    land_mask += continent_blob(lon_deg, lat_deg, 15.0, 50.0, 25.0, 15.0);
    land_mask += continent_blob(lon_deg, lat_deg, -5.0, 40.0, 10.0, 8.0);   // Iberia
    land_mask += continent_blob(lon_deg, lat_deg, 12.0, 42.0, 8.0, 6.0);    // Italy
    
    // Africa (centered around 20°, 5°)
    land_mask += continent_blob(lon_deg, lat_deg, 20.0, 5.0, 30.0, 35.0);
    land_mask += continent_blob(lon_deg, lat_deg, 35.0, 25.0, 15.0, 10.0);  // Middle East
    
    // Asia (massive, multiple blobs)
    land_mask += continent_blob(lon_deg, lat_deg, 90.0, 45.0, 50.0, 30.0);  // Central Asia
    land_mask += continent_blob(lon_deg, lat_deg, 105.0, 35.0, 25.0, 20.0); // China
    land_mask += continent_blob(lon_deg, lat_deg, 80.0, 22.0, 15.0, 12.0);  // India
    land_mask += continent_blob(lon_deg, lat_deg, 140.0, 38.0, 8.0, 12.0);  // Japan
    land_mask += continent_blob(lon_deg, lat_deg, 110.0, 0.0, 20.0, 15.0);  // Indonesia
    
    // Australia (centered around 135°, -25°)
    land_mask += continent_blob(lon_deg, lat_deg, 135.0, -25.0, 20.0, 15.0);
    
    // Antarctica (south pole)
    land_mask += continent_blob(lon_deg, lat_deg, 0.0, -80.0, 180.0, 10.0);
    
    // Greenland
    land_mask += continent_blob(lon_deg, lat_deg, -42.0, 72.0, 12.0, 10.0);
    
    // Combine with noise for natural coastlines
    let coast_noise = fbm_noise_2d(vec2<f32>(lon * 8.0, lat * 8.0), 3) * 0.15;
    let is_land = (land_mask + coast_noise) > 0.3;
    
    // ═══════════════════════════════════════════════════════════════
    // TERRAIN COLORING
    // ═══════════════════════════════════════════════════════════════
    
    // Ocean colors (deep blue with subtle variation)
    let ocean_deep = vec3<f32>(0.02, 0.08, 0.18);
    let ocean_shallow = vec3<f32>(0.05, 0.15, 0.30);
    let ocean_variation = fbm_noise_2d(vec2<f32>(lon * 3.0, lat * 3.0), 2);
    let ocean_color = mix(ocean_deep, ocean_shallow, ocean_variation * 0.5 + 0.3);
    
    // Land colors (varied by latitude and elevation)
    let elevation = terrain * 0.5 + 0.5;
    
    // Base land gradient (tropical → temperate → arctic)
    let tropical = vec3<f32>(0.15, 0.35, 0.12);   // Dark green
    let temperate = vec3<f32>(0.25, 0.40, 0.15);  // Medium green
    let boreal = vec3<f32>(0.20, 0.30, 0.18);     // Dark forest
    let tundra = vec3<f32>(0.35, 0.38, 0.35);     // Grey-green
    let ice = vec3<f32>(0.85, 0.88, 0.92);        // White-blue
    
    // Desert zones (around 25° latitude)
    let desert = vec3<f32>(0.55, 0.45, 0.30);     // Sandy
    let desert_factor = exp(-pow((abs(lat_deg) - 25.0) / 10.0, 2.0));
    
    // Mountain tint for high elevation
    let mountain = vec3<f32>(0.40, 0.35, 0.30);   // Brown-grey
    
    // Latitude-based biome selection
    let abs_lat = abs(lat_deg);
    var land_color: vec3<f32>;
    
    if (abs_lat < 15.0) {
        land_color = tropical;
    } else if (abs_lat < 35.0) {
        let t = (abs_lat - 15.0) / 20.0;
        land_color = mix(tropical, temperate, t);
        land_color = mix(land_color, desert, desert_factor * 0.6);
    } else if (abs_lat < 55.0) {
        let t = (abs_lat - 35.0) / 20.0;
        land_color = mix(temperate, boreal, t);
    } else if (abs_lat < 70.0) {
        let t = (abs_lat - 55.0) / 15.0;
        land_color = mix(boreal, tundra, t);
    } else {
        let t = (abs_lat - 70.0) / 20.0;
        land_color = mix(tundra, ice, clamp(t, 0.0, 1.0));
    }
    
    // Apply elevation variation
    land_color = mix(land_color, mountain, clamp(elevation - 0.6, 0.0, 0.4));
    
    // Add subtle terrain texture
    let texture_noise = fbm_noise_2d(vec2<f32>(lon * 20.0, lat * 20.0), 2);
    land_color = land_color * (0.9 + texture_noise * 0.2);
    
    // ═══════════════════════════════════════════════════════════════
    // FINAL COMPOSITING
    // ═══════════════════════════════════════════════════════════════
    
    var base_color = select(ocean_color, land_color, is_land);
    
    // ═══════════════════════════════════════════════════════════════
    // PROCEDURAL CLOUD LAYER (Photorealistic Earth enhancement)
    // ═══════════════════════════════════════════════════════════════
    
    // Animated cloud noise (slow drift over time)
    let cloud_time = camera.time * 0.02;  // Slow cloud movement
    let cloud_uv = vec2<f32>(lon + cloud_time, lat);
    
    // Multi-octave cloud noise
    var cloud = 0.0;
    cloud += 0.5 * fbm_noise_2d(cloud_uv * 3.0, 3);
    cloud += 0.3 * fbm_noise_2d(cloud_uv * 7.0 + vec2<f32>(cloud_time * 2.0, 0.0), 2);
    cloud += 0.2 * fbm_noise_2d(cloud_uv * 15.0, 2);
    cloud = (cloud + 1.0) * 0.5;  // Normalize to 0-1
    
    // Cloud density threshold (sparser clouds)
    let cloud_threshold = 0.55;
    let cloud_density = smoothstep(cloud_threshold, cloud_threshold + 0.15, cloud);
    
    // Tropical storm bands (ITCZ - Intertropical Convergence Zone)
    let itcz_factor = exp(-pow(lat_deg / 15.0, 2.0));  // Band around equator
    let storm_clouds = smoothstep(0.4, 0.6, cloud) * itcz_factor * 0.3;
    
    // Polar cloud caps
    let polar_factor = smoothstep(50.0, 75.0, abs(lat_deg));
    let polar_clouds = smoothstep(0.3, 0.5, cloud) * polar_factor * 0.4;
    
    // Combine cloud layers
    let total_cloud = clamp(cloud_density * 0.6 + storm_clouds + polar_clouds, 0.0, 0.85);
    
    // Cloud color (bright white with slight blue tint)
    let cloud_color = vec3<f32>(0.95, 0.97, 1.0);
    base_color = mix(base_color, cloud_color, total_cloud);
    
    // Atmospheric scattering (blue tint at edges)
    let view_dir = normalize(input.world_pos);
    let normal = normalize(input.normal);
    let fresnel = pow(1.0 - max(dot(-view_dir, normal), 0.0), 3.0);
    let atmosphere = vec3<f32>(0.4, 0.6, 1.0);
    base_color = mix(base_color, atmosphere, fresnel * 0.25);
    
    // Lighting (sun from upper right)
    let light_dir = normalize(vec3<f32>(0.6, 0.5, 0.8));
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let ambient = 0.25;
    let diffuse = 0.75 * n_dot_l;
    let lighting = ambient + diffuse;
    
    var color = vec4<f32>(base_color * lighting, 1.0);
    
    // Subtle grid overlay (30° spacing, very subtle)
    let lat_spacing = 30.0;
    let lon_spacing = 30.0;
    let grid_thickness = 0.02;
    
    let lat_mod = abs(fract(lat_deg / lat_spacing) - 0.5) * 2.0;
    let lon_mod = abs(fract(lon_deg / lon_spacing) - 0.5) * 2.0;
    
    if (lat_mod < grid_thickness || lon_mod < grid_thickness) {
        let grid_color = vec4<f32>(0.4, 0.5, 0.6, 1.0);
        color = mix(color, grid_color, 0.15);
    }
    
    return color;
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

// Elliptical continent blob
fn continent_blob(lon: f32, lat: f32, center_lon: f32, center_lat: f32, width: f32, height: f32) -> f32 {
    // Handle longitude wrapping
    var d_lon = lon - center_lon;
    if (d_lon > 180.0) { d_lon -= 360.0; }
    if (d_lon < -180.0) { d_lon += 360.0; }
    
    let d_lat = lat - center_lat;
    let dist = sqrt(pow(d_lon / width, 2.0) + pow(d_lat / height, 2.0));
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

// Simple satellite texture shader (Sprint 2)
// Uses bind group 1: binding 0 = texture, binding 1 = sampler
@group(1) @binding(0)
var sat_texture: texture_2d<f32>;

@group(1) @binding(1)
var sat_sampler: sampler;

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
    
    // Sample satellite texture
    let tex_color = textureSample(sat_texture, sat_sampler, vec2<f32>(u, v));
    
    // Calculate lighting with sun from upper-right
    // LOW AMBIENT + HIGH DIRECTIONAL = Terminator line (day/night contrast)
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

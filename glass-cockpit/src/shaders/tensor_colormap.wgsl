// Tensor Colormap Shader
// Scientific colormaps for tensor field visualization
// Constitutional compliance: Article V (float32 precision)

struct Uniforms {
    tensor_min: f32,
    tensor_max: f32,
    colormap_id: u32,  // 0=viridis, 1=plasma, 2=turbo, 3=inferno, 4=magma
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var tensor_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Viridis colormap (perceptually uniform, colorblind-friendly)
// Polynomial approximation from matplotlib
fn viridis(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.2670, 0.0049, 0.3294);
    let c1 = vec3<f32>(0.2777, 0.4692, 0.1062);
    let c2 = vec3<f32>(0.1534, 0.6790, -0.0575);
    let c3 = vec3<f32>(0.3304, 0.1836, 0.6371);
    
    let t2 = t * t;
    let t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3;
}

// Plasma colormap (high contrast, good for features)
fn plasma(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0505, 0.0298, 0.5280);
    let c1 = vec3<f32>(2.3583, 2.2660, -0.3298);
    let c2 = vec3<f32>(-2.8831, -8.7316, 3.1550);
    let c3 = vec3<f32>(1.8102, 7.0425, -3.0041);
    
    let t2 = t * t;
    let t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3;
}

// Turbo colormap (Google, high dynamic range)
fn turbo(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.1140, 0.0628, 0.2006);
    let c1 = vec3<f32>(6.7167, 3.1827, -7.4346);
    let c2 = vec3<f32>(-42.5683, -9.0072, 44.7236);
    let c3 = vec3<f32>(97.8906, 13.5578, -102.0815);
    let c4 = vec3<f32>(-82.6608, -10.0170, 84.0658);
    
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4;
}

// Inferno colormap (black-body radiation)
fn inferno(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0002, 0.0016, 0.0138);
    let c1 = vec3<f32>(0.8395, 1.3885, 0.0971);
    let c2 = vec3<f32>(1.3857, -2.2754, 3.3327);
    let c3 = vec3<f32>(-1.9923, 2.6542, -4.1361);
    
    let t2 = t * t;
    let t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3;
}

// Magma colormap (dark to light, high contrast)
fn magma(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0015, 0.0015, 0.0039);
    let c1 = vec3<f32>(0.6756, 1.4264, -0.3691);
    let c2 = vec3<f32>(1.6012, -3.6737, 5.6916);
    let c3 = vec3<f32>(-2.1578, 4.2625, -7.0914);
    
    let t2 = t * t;
    let t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3;
}

// Apply selected colormap
fn apply_colormap(t: f32) -> vec3<f32> {
    let clamped_t = clamp(t, 0.0, 1.0);
    
    switch uniforms.colormap_id {
        case 1u: { return plasma(clamped_t); }
        case 2u: { return turbo(clamped_t); }
        case 3u: { return inferno(clamped_t); }
        case 4u: { return magma(clamped_t); }
        default: { return viridis(clamped_t); }  // 0u or fallback
    }
}

@compute @workgroup_size(8, 8, 1)
fn colormap_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(tensor_texture);
    
    // Bounds check
    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }
    
    // Read tensor value (stored in red channel)
    let value = textureLoad(tensor_texture, coords, 0).r;
    
    // Check for special values (NaN, Inf)
    var color: vec3<f32>;
    if (isnan(value) || isinf(value)) {
        // Magenta sentinel for invalid values
        color = vec3<f32>(1.0, 0.0, 1.0);
    } else {
        // Normalize to [0, 1] using dynamic range
        let range = uniforms.tensor_max - uniforms.tensor_min;
        var normalized: f32;
        
        if (range > 0.0) {
            normalized = (value - uniforms.tensor_min) / range;
        } else {
            // Constant field - map to middle of colormap
            normalized = 0.5;
        }
        
        // Apply colormap
        color = apply_colormap(normalized);
    }
    
    // Write RGBA8 (alpha = 1.0 for opaque)
    textureStore(output_texture, coords, vec4<f32>(color, 1.0));
}

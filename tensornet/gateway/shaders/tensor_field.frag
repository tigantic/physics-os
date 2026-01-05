// Layer 1: Tensor Field - Fragment Shader
// ========================================
// Additive blending with plasma gradient
#version 450 core

in vec2 v_texcoord;
out vec4 frag_color;

uniform sampler2D scalar_field;

// Plasma gradient from valhalla_common.glsl (inlined for compatibility)
vec3 plasma_gradient(float t) {
    t = clamp(t, 0.0, 1.0);
    
    vec3 PLASMA_LOW = vec3(0.051, 0.031, 0.529);
    vec3 PLASMA_MID_LOW = vec3(0.416, 0.0, 0.659);
    vec3 PLASMA_MID = vec3(0.694, 0.165, 0.565);
    vec3 PLASMA_MID_HIGH = vec3(0.882, 0.392, 0.384);
    vec3 PLASMA_HIGH = vec3(0.988, 0.650, 0.212);
    
    if (t < 0.25) {
        float local_t = t / 0.25;
        return mix(PLASMA_LOW, PLASMA_MID_LOW, local_t);
    } else if (t < 0.5) {
        float local_t = (t - 0.25) / 0.25;
        return mix(PLASMA_MID_LOW, PLASMA_MID, local_t);
    } else if (t < 0.75) {
        float local_t = (t - 0.5) / 0.25;
        return mix(PLASMA_MID, PLASMA_MID_HIGH, local_t);
    } else {
        float local_t = (t - 0.75) / 0.25;
        return mix(PLASMA_MID_HIGH, PLASMA_HIGH, local_t);
    }
}

float opacity_map(float value, float min_alpha, float max_alpha) {
    return min_alpha + value * (max_alpha - min_alpha);
}

void main() {
    float value = texture(scalar_field, v_texcoord).r;
    
    // Map to plasma gradient
    vec3 color = plasma_gradient(value);
    
    // Opacity mapping: signal burns through noise
    float alpha = opacity_map(value, 0.1, 0.9);
    
    // Additive blending will be handled by OpenGL state
    frag_color = vec4(color, alpha);
}

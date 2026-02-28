// Layer 0: Geological Substrate - Fragment Shader
// ================================================
// Opaque base layer with 60% luminance darkening
#version 450 core

in vec2 v_texcoord;
out vec4 frag_color;

uniform sampler2D satellite_texture;
uniform float darkening_factor = 0.60;  // Photonic Discipline: 60% luminance

void main() {
    vec3 color = texture(satellite_texture, v_texcoord).rgb;
    
    // Darken to prevent vibration against energy overlays
    color *= darkening_factor;
    
    // Opaque base layer
    frag_color = vec4(color, 1.0);
}

// Layer 3: Sovereign Geometry - Fragment Shader
// ==============================================
// Premultiplied alpha for Cygnus Blue grid lines
#version 450 core

in vec3 v_color;
out vec4 frag_color;

uniform float grid_opacity = 0.5;

void main() {
    // Cygnus Blue: #00E5FF → vec3(0.0, 0.898, 1.0)
    vec3 color = v_color.rgb;
    
    // Premultiplied alpha: color already multiplied by alpha
    frag_color = vec4(color * grid_opacity, grid_opacity);
}

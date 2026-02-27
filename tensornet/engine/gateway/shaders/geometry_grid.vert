// Layer 3: Sovereign Geometry - Vertex Shader
// ============================================
#version 450 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

out vec3 v_color;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}

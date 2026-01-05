// Layer 0: Geological Substrate - Vertex Shader
// ==============================================
#version 450 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;

out vec2 v_texcoord;

uniform mat4 mvp;  // Model-View-Projection matrix

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_texcoord = in_texcoord;
}

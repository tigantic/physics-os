// simple_quad.vert
// Full-screen quad vertex shader for implicit synthesis

#version 430 core

layout(location = 0) in vec2 a_position;  // [-1, 1] NDC quad

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}

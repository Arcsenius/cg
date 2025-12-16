#version 450

layout(location = 0) in vec3 in_position;

layout(binding = 0) uniform LightSpaceUniforms {
    mat4 light_space_matrix;
} light_space;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad0;
} model_data;

void main() {
    gl_Position = light_space.light_space_matrix * model_data.model * vec4(in_position, 1.0);
}
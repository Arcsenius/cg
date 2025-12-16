#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

struct DirectionalLight {
    vec3 direction;
    float _pad0;
    vec3 ambient;
    float _pad1;
    vec3 diffuse;
    float _pad2;
    vec3 specular;
    float _pad3;
};

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    DirectionalLight directional_light;
    uint point_light_count;
    uint spot_light_count;
    float _pad1;
    float _pad2;
    mat4 light_space_matrix;
} scene;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad0;
} model;

layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec4 frag_position_light_space;

void main() {
    vec4 world_position = model.model * vec4(in_position, 1.0);
    gl_Position = scene.view_projection * world_position;
    
    frag_position = world_position.xyz;
    frag_normal = mat3(model.model) * in_normal;
    frag_uv = in_uv;
    frag_position_light_space = scene.light_space_matrix * world_position;
}
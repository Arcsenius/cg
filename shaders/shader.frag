#version 450
#extension GL_ARB_separate_shader_objects : enable

struct GlobalLight {
    vec3 ambient_color; float pad1;
    vec3 directional_color; float pad2;
    vec3 directional_direction; float pad3;
};

struct PointLight {
    vec3 position; float pad1;
    vec3 color; float pad2;
};

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 outColor;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 view_inverse;
    vec3 camera_position; float pad0;
    GlobalLight global_light;
    uint num_point_lights; float pad_align[3];
} scene_ubo;

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float pad1;
} model_ubo;

layout(binding = 2, std430) readonly buffer PointLightsBuffer {
    PointLight lights[];
} light_ssbo;

vec3 calculateBlinnPhong(
    vec3 lightDir,
    vec3 lightColor,
    vec3 camPos,
    vec3 fragPos,
    vec3 normal,
    float attenuation)
{
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(camPos - fragPos);

    // Diffuse component
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = model_ubo.albedo_color * lightColor * diff;

    // Specular component (Blinn-Phong)
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), model_ubo.shininess);
    vec3 specular = model_ubo.specular_color * lightColor * spec;

    return (diffuse + specular) * attenuation;
}

void main() {
    vec3 N = normalize(f_normal);
    
    // Ambient lighting
    vec3 total_lighting = model_ubo.albedo_color * scene_ubo.global_light.ambient_color;

    // Directional light
    vec3 dir_L = -normalize(scene_ubo.global_light.directional_direction);
    total_lighting += calculateBlinnPhong(
        dir_L,
        scene_ubo.global_light.directional_color,
        scene_ubo.camera_position,
        f_position,
        N,
        1.0);

    // Point lights with inverse square law attenuation
    for (uint i = 0; i < scene_ubo.num_point_lights; ++i) {
        PointLight light = light_ssbo.lights[i];
        vec3 L_vec = light.position - f_position;
        float distance = length(L_vec);
        
        // Inverse square law: 1.0 / (distance * distance)
        float attenuation = 1.0 / (distance * distance);
        
        total_lighting += calculateBlinnPhong(
            L_vec,
            light.color,
            scene_ubo.camera_position,
            f_position,
            N,
            attenuation);
    }

    outColor = vec4(total_lighting, 1.0);
}

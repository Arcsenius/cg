#version 450

layout(location = 0) in vec3 frag_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 frag_position_light_space;

layout(location = 0) out vec4 out_color;

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
    float _pad1[2];
    mat4 light_space_matrix;
} scene;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad0;
} material;

layout(binding = 2) uniform sampler2D material_texture;
layout(binding = 3) uniform sampler2DShadow shadow_map;
layout(binding = 4) uniform sampler2D shadow_map_raw;

float calculateShadow(vec4 frag_pos_light_space) {
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;
    
    if (proj_coords.z > 1.0 || proj_coords.z < 0.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }
    
    float bias = 0.005;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map_raw, 0));
    
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texel_size;
            shadow += texture(shadow_map, vec3(proj_coords.xy + offset, proj_coords.z - bias));
        }
    }
    shadow /= 9.0;
    
    return shadow;
}

vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 view_dir) {
    vec3 light_dir = normalize(-light.direction);
    vec3 ambient = light.ambient * material.albedo_color;
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * material.albedo_color;
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * material.specular_color;
    return ambient + diffuse + specular;
}

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(scene.view_position - frag_position);
    vec3 result = vec3(0.0);
    float shadow = calculateShadow(frag_position_light_space);
    vec3 ambient = scene.directional_light.ambient * material.albedo_color;
    result += ambient;
    vec3 light_dir = normalize(-scene.directional_light.direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = scene.directional_light.diffuse * diff * material.albedo_color;
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = scene.directional_light.specular * spec * material.specular_color;
    result += (diffuse + specular) * shadow;
    vec3 texture_color = texture(material_texture, frag_uv).rgb;
    result *= texture_color;
    out_color = vec4(result, 1.0);
}
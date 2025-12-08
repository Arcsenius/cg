#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

struct LightColors {
    vec3 ambient; float _pad0;
    vec3 diffuse; float _pad1;
    vec3 specular; float _pad2;
};

struct DirectionalLight {
    vec3 direction;
    float intensity;
    LightColors colors;
};

struct AmbientLight {
    vec3 color;
    float intensity;
    vec3 specular_color;
    float shininess;
};

struct PointLight {
    vec3 position; float _pad0;
    LightColors colors;
    float linear;
    float quadratic;
    float _pad1;
    float _pad2;
};

struct SpotLight {
    PointLight point_light;
    vec3 direction;
    float cut_off;
    float outer_cut_off;
    float _pad1;
    float _pad2;
};

// Binding 0: Scene Uniforms
layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position; // <-- ИЗМЕНЕНО НА vec4 ДЛЯ ВЫРАВНИВАНИЯ
    uint num_point_lights;
    uint num_spot_lights;
    float _pad_align[2];  // Явное выравнивание
    DirectionalLight directional_light;
    AmbientLight ambient_light;
} scene;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
} material;

layout(std140, binding = 2) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointLights;

layout(std140, binding = 3) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotLights;

layout(binding = 4) uniform sampler2D texSide;
layout(binding = 5) uniform sampler2D texTop;
layout(binding = 6) uniform sampler2D texBottom;

vec3 calculateBlinnPhong(LightColors colors, vec3 lightDir, vec3 viewDir, vec3 normal, float attenuation, float intensity, vec3 texColor) {
    vec3 ambient = colors.ambient * texColor;
    
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * colors.diffuse * texColor;
    
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    vec3 specular = spec * colors.specular * material.specular_color;
    
    return (ambient + diffuse + specular) * attenuation * intensity;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.camera_position.xyz - fragPos); // берем .xyz от vec4
    
    vec4 sampledColor;
    if (N.y > 0.9) {
        sampledColor = texture(texTop, fragUV);
    } else if (N.y < -0.9) {
        sampledColor = texture(texBottom, fragUV);
    } else {
        sampledColor = texture(texSide, fragUV);
    }
    
    vec3 albedo = sampledColor.rgb; 

    vec3 totalLight = vec3(0.0);

    // 1. Ambient
    totalLight += scene.ambient_light.color * scene.ambient_light.intensity * albedo;

    // 2. Directional
    vec3 L_dir = normalize(-scene.directional_light.direction);
    totalLight += calculateBlinnPhong(scene.directional_light.colors, L_dir, V, N, 1.0, scene.directional_light.intensity, albedo);

    // 3. Point Lights
    for(uint i = 0; i < scene.num_point_lights; ++i) {
        PointLight light = pointLights.lights[i];
        vec3 L = normalize(light.position - fragPos);
        float dist = length(light.position - fragPos);
        float attenuation = 1.0 / (1.0 + light.linear * dist + light.quadratic * (dist * dist));
        totalLight += calculateBlinnPhong(light.colors, L, V, N, attenuation, 1.0, albedo);
    }

    // 4. Spot Lights
    for(uint i = 0; i < scene.num_spot_lights; ++i) {
        SpotLight sLight = spotLights.lights[i];
        PointLight light = sLight.point_light;
        vec3 L = normalize(light.position - fragPos);
        float dist = length(light.position - fragPos);
        float attenuation = 1.0 / (1.0 + light.linear * dist + light.quadratic * (dist * dist));
        float theta = dot(L, normalize(-sLight.direction));
        float epsilon = sLight.cut_off - sLight.outer_cut_off;
        float intensity = clamp((theta - sLight.outer_cut_off) / epsilon, 0.0, 1.0);
        totalLight += calculateBlinnPhong(light.colors, L, V, N, attenuation, intensity, albedo);
    }

    outColor = vec4(totalLight, 1.0);
}
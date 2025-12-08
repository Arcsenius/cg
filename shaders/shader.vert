#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    // Остальное не нужно в вертексном
} scene;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
} mesh;

void main() {
    vec4 worldPosition = mesh.model * vec4(inPosition, 1.0);
    fragPos = worldPosition.xyz;
    
    // Важно: нормали должны быть нормализованы после интерполяции, 
    // но здесь мы просто передаем их дальше.
    fragNormal = mat3(mesh.normal_matrix) * inNormal;
    
    fragUV = inUV;
    
    gl_Position = scene.view_projection * worldPosition;
}
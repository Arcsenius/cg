#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

// Binding 0: Scene Uniforms (нам здесь нужна только матрица проекции)
layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    // Остальные поля здесь не нужны, GLSL сам найдет смещение
} scene;

// Binding 1: Model Uniforms
layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
} mesh;

void main() {
    // Вычисляем мировую позицию вершины
    vec4 worldPosition = mesh.model * vec4(inPosition, 1.0);
    fragPos = worldPosition.xyz;

    // Вычисляем нормаль с учетом вращения модели
    // mat3(normal_matrix) убирает перенос, оставляя только вращение/масштаб
    fragNormal = mat3(mesh.normal_matrix) * inNormal;
    
    fragUV = inUV;

    // Позиция на экране
    gl_Position = scene.view_projection * worldPosition;
}
#version 450

struct GlobalLight {
    vec3 ambient_color; float pad1;
    vec3 directional_color; float pad2;
    vec3 directional_direction; float pad3;
};

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 view_inverse;
	vec3 camera_position; float pad0;
	GlobalLight global_light;
	uint num_point_lights; float pad_align[3];
} scene_ubo;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec3 albedo_color; float shininess;
	vec3 specular_color; float pad1;
} model_ubo;

void main() {
	vec4 position = model_ubo.model * vec4(v_position, 1.0f);
	vec4 normal = model_ubo.normal_matrix * vec4(v_normal, 0.0f);

	gl_Position = scene_ubo.view_projection * position;

	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv;
}

#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 4; // Максимальное число точечных/прожекторных источников

// --- НОВЫЕ СТРУКТУРЫ ОСВЕЩЕНИЯ ---

struct PointLight {
	// Attenuation constants are typically 1.0, 0.7, 1.8 for medium range
	veekay::vec3 position; float attenuation_const = 1.0f;
	veekay::vec3 color; float attenuation_linear = 0.7f;
	float attenuation_quad = 1.8f; float _pad[3]; // Затухание по обратному квадрату
};

struct GlobalLight {
	veekay::vec3 ambient_color = {0.05f, 0.05f, 0.05f}; float _pad0;
	veekay::vec3 directional_color = {0.8f, 0.8f, 0.8f}; float _pad1;
	// Direction must be normalized, defined in world space
	veekay::vec3 directional_direction = veekay::vec3::normalized({-0.5f, -1.0f, -0.2f}); float _pad2;
};

// --- СТРУКТУРЫ ДЛЯ GPU ---

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::mat4 view_inverse; // Для расчета направления/позиции камеры
	veekay::vec3 camera_position; float _pad0;
	GlobalLight global_light;
	uint32_t num_point_lights; float _pad_align[3];
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::mat4 normal_matrix; // Для преобразования нормалей
	veekay::vec3 albedo_color; float shininess; // Альбедо и блеск
	veekay::vec3 specular_color; float _pad1;    // Цвет блика
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

Mesh createConeMesh(uint32_t segments, float radius, float height);

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	
	// --- МАТЕРИАЛЫ ---
	veekay::vec3 albedo_color = {1.0f, 1.0f, 1.0f};
	veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f; // Параметр блеска

	// --- АНИМАЦИЯ ---
	veekay::vec3 rotation_axis = {0.0f, 1.0f, 0.0f};
	float rotation_speed = 0.0f;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;
    constexpr static float mouse_sensitivity = 0.005f;
    constexpr static float movement_speed = 3.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {}; // Pitch (X) and Yaw (Y)

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

    void processInput(float delta_time);
	veekay::mat4 view() const;
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// --- СЦЕНА И ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
inline namespace {
	Camera camera{
		.position = {0.0f, 0.0f, -3.0f}
	};

	std::vector<Model> models;
	std::array<PointLight, max_point_lights> point_lights;
    GlobalLight global_light_params;
    
    // Переменные для управления камерой
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool first_mouse = true;
}

// --- VULKAN ОБЪЕКТЫ ---
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* light_ssbo; // НОВЫЙ БУФЕР: Shader Storage Buffer Object

	Mesh cone_mesh;

	float rotation_speed_multiplier = 1.0f;
	double previous_time = 0.0;
	bool has_previous_time = false;
}

// --- РЕАЛИЗАЦИЯ ТРАНСФОРМАЦИЙ И КАМЕРЫ ---

veekay::mat4 Transform::matrix() const {
	auto rotation_matrix = [](const veekay::vec3& axis, float angle) {
		if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) {
			return veekay::mat4::identity();
		}
		return veekay::mat4::rotation(axis, angle);
	};

	auto translation = veekay::mat4::translation(position);
	auto scaling = veekay::mat4::scaling(scale);

	auto rotate_x = rotation_matrix({1.0f, 0.0f, 0.0f}, rotation.x);
	auto rotate_y = rotation_matrix({0.0f, 1.0f, 0.0f}, rotation.y);
	auto rotate_z = rotation_matrix({0.0f, 0.0f, 1.0f}, rotation.z);

	return translation * rotate_z * rotate_y * rotate_x * scaling;
}

Mesh createConeMesh(uint32_t segments, float radius, float height) {
	segments = std::max<uint32_t>(3, segments);
	const float half_height = height * 0.5f;
	const float slope = radius / height;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	vertices.reserve(segments * 4 + 2);
	indices.reserve(segments * 6);

	std::vector<uint32_t> apex_indices(segments);
	std::vector<uint32_t> rim_indices(segments);

	for (uint32_t i = 0; i < segments; ++i) {
		const float t = static_cast<float>(i) / static_cast<float>(segments);
		const float angle = t * 2.0f * static_cast<float>(M_PI);
		const float cos_angle = std::cos(angle);
		const float sin_angle = std::sin(angle);

		const veekay::vec3 normal = veekay::vec3::normalized({cos_angle, slope, sin_angle});

		const uint32_t apex_index = static_cast<uint32_t>(vertices.size());
		apex_indices[i] = apex_index;
		vertices.push_back({
			.position = {0.0f, half_height, 0.0f},
			.normal = normal,
			.uv = {t, 1.0f}
		});

		const uint32_t rim_index = static_cast<uint32_t>(vertices.size());
		rim_indices[i] = rim_index;
		vertices.push_back({
			.position = {radius * cos_angle, -half_height, radius * sin_angle},
			.normal = normal,
			.uv = {t, 0.0f}
		});
	}

	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t next = (i + 1) % segments;
		indices.push_back(apex_indices[i]);
		indices.push_back(rim_indices[i]);
		indices.push_back(rim_indices[next]);
	}

	const uint32_t base_center_index = static_cast<uint32_t>(vertices.size());
	vertices.push_back({
		.position = {0.0f, -half_height, 0.0f},
		.normal = {0.0f, -1.0f, 0.0f},
		.uv = {0.5f, 0.5f}
	});

	std::vector<uint32_t> base_ring_indices(segments);
	for (uint32_t i = 0; i < segments; ++i) {
		const float t = static_cast<float>(i) / static_cast<float>(segments);
		const float angle = t * 2.0f * static_cast<float>(M_PI);
		const float cos_angle = std::cos(angle);
		const float sin_angle = std::sin(angle);

		const uint32_t index = static_cast<uint32_t>(vertices.size());
		base_ring_indices[i] = index;
		vertices.push_back({
			.position = {radius * cos_angle, -half_height, radius * sin_angle},
			.normal = {0.0f, -1.0f, 0.0f},
			.uv = {0.5f + 0.5f * cos_angle, 0.5f + 0.5f * sin_angle}
		});
	}

	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t next = (i + 1) % segments;
		indices.push_back(base_center_index);
		indices.push_back(base_ring_indices[next]);
		indices.push_back(base_ring_indices[i]);
	}

	Mesh mesh{};
	mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex),
	                                                  vertices.data(),
	                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t),
	                                                 indices.data(),
	                                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	mesh.indices = static_cast<uint32_t>(indices.size());

	return mesh;
}

veekay::mat4 Camera::view() const {
    // Вращение: Y (Yaw) * X (Pitch). Z (Roll) обычно не используется в FPS-камерах.
    veekay::mat4 pitch = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    veekay::mat4 yaw = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    
    veekay::mat4 R = yaw * pitch; // Порядок важен: Yaw сначала, Pitch потом.

    // Матрица Вида = (R * T).inverse() = T.inverse() * R.inverse()
    // R.inverse() == R.transpose() для ортонормальной матрицы вращения
    return veekay::mat4::transpose(R) * veekay::mat4::translation(-position);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    // Vulkan использует правые системы координат, но инвертированные Y и Z (из-за GLM legacy).
    // veekay::mat4::projection, вероятно, уже учитывает это.
	return projection * view();
}

void Camera::processInput(float delta_time) {
    // Взаимодействие с клавиатурой (WASD)
    veekay::vec3 front = {
        std::sin(rotation.y) * std::cos(rotation.x),
        std::sin(rotation.x),
        std::cos(rotation.y) * std::cos(rotation.x)
    };
    front = veekay::vec3::normalized(front);

    veekay::vec3 up = {0.0f, 1.0f, 0.0f};
    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, up));

    float move_speed = movement_speed * delta_time;

    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) {
        position += front * move_speed;
    }
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) {
        position -= front * move_speed;
    }
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) {
        position -= right * move_speed;
    }
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) {
        position += right * move_speed;
    }

    // Взаимодействие с мышью
    veekay::vec2 cursor_position = veekay::input::mouse::cursorPosition();
    double current_x = cursor_position.x;
    double current_y = cursor_position.y;

    if (first_mouse) {
        last_mouse_x = current_x;
        last_mouse_y = current_y;
        first_mouse = false;
    }

    float x_offset = static_cast<float>(current_x - last_mouse_x);
    float y_offset = static_cast<float>(last_mouse_y - current_y); // Обратный Y для инверсии
    
    last_mouse_x = current_x;
    last_mouse_y = current_y;

    x_offset *= mouse_sensitivity;
    y_offset *= mouse_sensitivity;

    rotation.y += x_offset; // Yaw
    rotation.x += y_offset; // Pitch

    // Ограничение Pitch (поворот вверх/вниз)
    const float pitch_limit = 0.9f * static_cast<float>(M_PI_2);
    rotation.x = std::clamp(rotation.x, -pitch_limit, pitch_limit);
}

// --- УТИЛИТЫ VULKAN ---

VkShaderModule loadShaderModule(const char* path) {
    // ... (функция loadShaderModule остается без изменений) ...
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

// --- ИНИЦИАЛИЗАЦИЯ ---

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;

    veekay::input::mouse::setCaptured(true);

    // --- НАСТРОЙКА ИСТОЧНИКОВ СВЕТА ---
    global_light_params.directional_color = {0.8f, 0.8f, 0.8f};
    global_light_params.ambient_color = {0.1f, 0.1f, 0.1f};

    // Точечные источники
    point_lights[0] = {
        .position = {2.0f, 1.0f, 0.0f},
        .attenuation_const = 1.0f,
        .color = {1.0f, 0.5f, 0.5f}, // Красный
        .attenuation_linear = 0.4f,
        .attenuation_quad = 0.2f
    };
    point_lights[1] = {
        .position = {-2.0f, 1.0f, 0.0f},
        .attenuation_const = 1.0f,
        .color = {0.5f, 0.5f, 1.0f}, // Синий
        .attenuation_linear = 0.4f,
        .attenuation_quad = 0.2f
    };
    // Остальные можно оставить выключенными или настроить как прожекторы

	{ // НОВЫЙ VULKAN PIPELINE: Поддержка SSBO
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

        // ... (stage infos, vertex input setup remains the same) ...
        
		VkPipelineShaderStageCreateInfo stage_infos[2];
		stage_infos[0] = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader_module, .pName = "main" };
		stage_infos[1] = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader_module, .pName = "main" };
        
        // (Остальные настройки pipeline state, input state и т.д. опущены для краткости, они остались прежними)
        
        // ... (Vertex Input and Assembly setup) ...
        VkVertexInputBindingDescription buffer_binding{ .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
		VkVertexInputAttributeDescription attributes[] = {
			{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
			{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
			{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
		};
        VkPipelineVertexInputStateCreateInfo input_state_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &buffer_binding, .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes };
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        VkPipelineRasterizationStateCreateInfo raster_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_BACK_BIT, .frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f };
        VkPipelineMultisampleStateCreateInfo sample_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        VkViewport viewport{ .x = 0.0f, .y = 0.0f, .width = static_cast<float>(veekay::app.window_width), .height = static_cast<float>(veekay::app.window_height), .minDepth = 0.0f, .maxDepth = 1.0f };
		VkRect2D scissor{ .offset = {0, 0}, .extent = {veekay::app.window_width, veekay::app.window_height} };
		VkPipelineViewportStateCreateInfo viewport_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };
        VkPipelineDepthStencilStateCreateInfo depth_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL };
        VkPipelineColorBlendAttachmentState attachment_info{ .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
        VkPipelineColorBlendStateCreateInfo blend_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .logicOpEnable = false, .attachmentCount = 1, .pAttachments = &attachment_info };

		{ // Дескрипторный пул: увеличено количество для SSBO
			VkDescriptorPoolSize pools[] = {
				{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8 },
				{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 8 },
                { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8 }, // SSBO
				{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 8 }
			};
			
			VkDescriptorPoolCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = sizeof(pools) / sizeof(pools[0]), .pPoolSizes = pools };

			if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		{ // Layout: добавлена привязка 2 для SSBO
			VkDescriptorSetLayoutBinding bindings[] = {
				{ // Binding 0: Scene Uniforms (VP, Camera Pos, Global Lights)
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{ // Binding 1: Model Uniforms (Dynamic)
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
                { // Binding 2: Light SSBO (Point Lights Array)
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                }
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}
        
        // ... (Allocate Descriptor Set) ...
		{
			VkDescriptorSetAllocateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout };
			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}


		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
		};

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

    // --- СОЗДАНИЕ БУФЕРОВ ---
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        
    light_ssbo = new veekay::graphics::Buffer(
        sizeof(PointLight) * max_point_lights,
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); // Использование STORAGE_BUFFER

	{ // Обновление дескрипторов (добавлена привязка 2)
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
            {
                .buffer = light_ssbo->buffer,
                .offset = 0,
                .range = sizeof(PointLight) * max_point_lights,
            },
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 1, .descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffer_infos[1],
			},
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 2, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_infos[2],
            },
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// --- МЕШИ И МОДЕЛИ ---
	{
		constexpr uint32_t cone_segments = 48;
		const float cone_radius = 0.6f;
		const float cone_height = 1.4f;
		cone_mesh = createConeMesh(cone_segments, cone_radius, cone_height);
	}

	{
		struct ConeConfig {
			veekay::vec3 position;
			veekay::vec3 scale;
			veekay::vec3 rotation_axis;
			float rotation_speed;
		};

		const std::array<ConeConfig, 5> cones = {{
			{{-2.3f, -0.7f, -2.2f}, {0.9f, 0.9f, 0.9f}, {0.0f, 1.0f, 0.0f}, 0.6f},
			{{-0.8f, -0.7f, -1.0f}, {0.7f, 1.1f, 0.7f}, {0.3f, 1.0f, 0.1f}, 0.9f},
			{{1.6f, -0.7f, -1.4f}, {1.0f, 0.8f, 1.0f}, {0.0f, 1.0f, 0.4f}, 1.3f},
			{{0.4f, -0.7f, -2.6f}, {0.6f, 1.2f, 0.6f}, {0.2f, 1.0f, 0.3f}, 0.8f},
			{{2.2f, -0.7f, -0.8f}, {1.1f, 0.9f, 0.9f}, {0.4f, 1.0f, 0.0f}, 1.6f},
		}};

		models.clear();
		models.reserve(cones.size());

		for (const auto& cone : cones) {
			Model instance;
			instance.mesh = cone_mesh;
			instance.transform.position = cone.position;
			instance.transform.scale = cone.scale;
			instance.albedo_color = {0.95f, 0.55f, 0.2f};
            instance.specular_color = {1.0f, 1.0f, 1.0f};
            instance.shininess = 32.0f; 
			instance.rotation_axis = veekay::vec3::normalized(cone.rotation_axis);
			instance.rotation_speed = cone.rotation_speed;

			models.emplace_back(instance);
		}

		previous_time = 0.0;
		has_previous_time = false;
	}
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	delete cone_mesh.index_buffer;
	delete cone_mesh.vertex_buffer;

    delete light_ssbo; // Уничтожаем новый буфер
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

// --- ЦИКЛ ОБНОВЛЕНИЯ ---

void update(double time) {
	float delta_time = 0.0f;
	if (has_previous_time) {
		delta_time = static_cast<float>(time - previous_time);
	} else {
		has_previous_time = true;
	}
	previous_time = time;
	delta_time = std::max(delta_time, 0.0f);
    
    // 1. Управление Камерой
    camera.processInput(delta_time);

	// 2. Управление UI (Освещение)
	ImGui::Begin("Lighting Controls");
	ImGui::Text("Global Animation");
	ImGui::SliderFloat("Rotation speed", &rotation_speed_multiplier, 0.0f, 5.0f);
	
    ImGui::Separator();
    ImGui::Text("Ambient and Directional Light");
    ImGui::ColorEdit3("Ambient Color", &global_light_params.ambient_color.x);
    ImGui::ColorEdit3("Directional Color", &global_light_params.directional_color.x);
    ImGui::SliderFloat3("Directional Direction", &global_light_params.directional_direction.x, -1.0f, 1.0f);
    global_light_params.directional_direction = veekay::vec3::normalized(global_light_params.directional_direction);

    ImGui::Separator();
    ImGui::Text("Point Light 1 (Red)");
    ImGui::ColorEdit3("L1 Color", &point_lights[0].color.x);
    ImGui::SliderFloat3("L1 Position", &point_lights[0].position.x, -5.0f, 5.0f);
    ImGui::SliderFloat("L1 Const Atten", &point_lights[0].attenuation_const, 0.0f, 10.0f);
    ImGui::SliderFloat("L1 Linear Atten", &point_lights[0].attenuation_linear, 0.0f, 5.0f);
    ImGui::SliderFloat("L1 Quad Atten", &point_lights[0].attenuation_quad, 0.0f, 5.0f);

    ImGui::Text("Point Light 2 (Blue)");
    ImGui::ColorEdit3("L2 Color", &point_lights[1].color.x);
    ImGui::SliderFloat3("L2 Position", &point_lights[1].position.x, -5.0f, 5.0f);
    ImGui::End();


	// 3. Анимация Моделей
	const float scaled_delta = delta_time * rotation_speed_multiplier;
	for (Model& model : models) {
		const float angle_delta = model.rotation_speed * scaled_delta;
		model.transform.rotation += model.rotation_axis * angle_delta;
	}

	// 4. Обновление Uniform Buffers на CPU
	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    
    veekay::mat4 view_matrix = camera.view();
    veekay::mat4 view_projection_matrix = camera.view_projection(aspect_ratio);

	uint32_t active_point_lights = 0;
	for (const auto& light : point_lights) {
		if (veekay::vec3::squaredLength(light.color) > 0.0f) {
			++active_point_lights;
		}
	}
	active_point_lights = std::min<uint32_t>(active_point_lights, max_point_lights);

	SceneUniforms scene_uniforms{
		.view_projection = view_projection_matrix,
        .view_inverse = veekay::mat4::inverse(view_matrix), // Используется для World Space
        .camera_position = camera.position,
        ._pad0 = 0.0f,
        .global_light = global_light_params,
        .num_point_lights = active_point_lights, // Рисуем только доступные источники
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];
        
        uniforms.model = model.transform.matrix();
        // Матрица нормалей: инверсия и транспонирование Модельной матрицы
        // Если Model Matrix не содержит не-универсального масштабирования, можно использовать просто Model Matrix.
        // Здесь используется упрощенный подход, но по правилам Blinn-Phong нужна инвертированная транспонированная матрица 3x3.
        uniforms.normal_matrix = veekay::mat4::transpose(veekay::mat4::inverse(uniforms.model)); 
        
		uniforms.albedo_color = model.albedo_color;
        uniforms.shininess = model.shininess;
        uniforms.specular_color = model.specular_color;
	}

	// 5. Копирование данных в GPU
	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
    
    // Копирование SSBO (Point Lights)
    memcpy(light_ssbo->mapped_region, point_lights.data(), sizeof(PointLight) * max_point_lights);
    
	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];
		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

    // ... (Начало командного буфера и рендер-прохода остается прежним) ...

	{ 
		VkCommandBufferBeginInfo info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, };
		vkBeginCommandBuffer(cmd, &info);
	}

	{ 
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = { .extent = { veekay::app.window_width, veekay::app.window_height } },
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

        // ... (Привязка буферов геометрии) ...

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
        // Динамический оффсет применяется только к Binding 1
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
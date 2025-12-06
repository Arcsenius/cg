#include "veekay/input.hpp"
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
#include <lodepng.h>

// Если M_PI не определен (зависит от компилятора), определим его
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;

struct Vertex {
  veekay::vec3 position;
  veekay::vec3 normal;
  veekay::vec2 uv;
};

struct GlobalLight {
  veekay::vec3 ambient_color; float _pad1;
  veekay::vec3 directional_color; float _pad2;
  veekay::vec3 directional_direction; float _pad3;
};

struct SceneUniforms {
  veekay::mat4 view_projection;
  veekay::mat4 view_inverse;
  veekay::vec3 camera_position; float _pad0;
  GlobalLight global_light;
  uint32_t num_point_lights; float _pad_align[3];
};

struct PointLight {
  veekay::vec3 position; float _pad1;
  veekay::vec3 color; float _pad2;
};

struct ModelUniforms {
  veekay::mat4 model;
  veekay::mat4 normal_matrix;
  veekay::vec3 albedo_color; float shininess;
  veekay::vec3 specular_color; float _pad1;
};

struct Mesh {
  veekay::graphics::Buffer* vertex_buffer;
  veekay::graphics::Buffer* index_buffer;
  uint32_t indices;
};

struct Transform {
  veekay::vec3 position = {};
  veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
  veekay::vec3 rotation = {};

  veekay::mat4 matrix() const;
};

struct Model {
  Mesh mesh;
  Transform transform;
  veekay::vec3 albedo_color;
  veekay::vec3 rotation_axis = {0.0f, 1.0f, 0.0f};
  float rotation_speed = 0.0f;
};

// --- ИСПРАВЛЕННЫЙ КЛАСС CAMERA ---
struct Camera {
  constexpr static float default_fov = 70.0f;
  constexpr static float default_near_plane = 0.01f;
  constexpr static float default_far_plane = 100.0f;

  veekay::vec3 position = {};
  veekay::vec3 rotation = {}; // x = pitch (наклон), y = yaw (поворот)

  float fov = default_fov;
  float near_plane = default_near_plane;
  float far_plane = default_far_plane;

  // Вспомогательная функция для получения вектора "вперед" (нужна для управления WASD)
  veekay::vec3 getFront() const {
      const float pitch = rotation.x;
      const float yaw = rotation.y;

      const veekay::vec3 front = {
          std::cos(pitch) * std::sin(yaw),
          std::sin(pitch),
          -std::cos(pitch) * std::cos(yaw)
      };
      return veekay::vec3::normalized(front);
  }

  // ИСПРАВЛЕНИЕ: Используем вашу оригинальную логику через inverse,
  // так как veekay::mat4::lookAt отсутствует в вашей версии библиотеки.
  veekay::mat4 view() const {
      auto rotation_matrix = [](const veekay::vec3& axis, float angle) {
          if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) {
              return veekay::mat4::identity();
          }
          return veekay::mat4::rotation(axis, angle);
      };

      auto translation = veekay::mat4::translation(position);
      
      // Порядок вращения для FPS камеры: Сначала Yaw (Y), потом Pitch (X)
      // Вращение по Z обычно не используется в FPS камерах
      auto rotate_x = rotation_matrix({1.0f, 0.0f, 0.0f}, rotation.x);
      auto rotate_y = rotation_matrix({0.0f, 1.0f, 0.0f}, rotation.y);
      
      // Матрица мира камеры = Перемещение * Поворот Y * Поворот X
      veekay::mat4 camera_world_matrix = translation * rotate_y * rotate_x;

      // Матрица вида - это обратная матрица мира камеры
      return veekay::mat4::inverse(camera_world_matrix);
  }

  veekay::mat4 view_projection(float aspect_ratio) const;
};
// ---------------------------------

inline namespace {
  Camera camera{
    .position = {0.0f, -0.5f, -3.0f}
  };

  std::vector<Model> models;
}

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
  veekay::graphics::Buffer* point_lights_buffer;

  Mesh cone_mesh;

  float rotation_speed_multiplier = 1.0f;
  double previous_time = 0.0;
  bool has_previous_time = false;

  const veekay::vec3 cone_color = {0.95f, 0.55f, 0.2f};
}

Mesh createConeMesh(uint32_t segments, float radius, float height) {
  segments = std::max(segments, 3u);

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  vertices.reserve(1 + segments + 1 + segments);
  indices.reserve(segments * 6);

  vertices.push_back(Vertex{
    .position = {0.0f, height, 0.0f},
    .normal = {0.0f, 1.0f, 0.0f},
    .uv = {0.5f, 0.0f},
  });

  const float circumference = 2.0f * float(M_PI);

  for (uint32_t i = 0; i < segments; ++i) {
    float angle = circumference * (float(i) / float(segments));
    float x = radius * std::cos(angle);
    float z = radius * std::sin(angle);

    veekay::vec3 normal = veekay::vec3::normalized({
      x,
      radius / height,
      z
    });

    vertices.push_back(Vertex{
      .position = {x, 0.0f, z},
      .normal = normal,
      .uv = {float(i) / float(segments), 1.0f},
    });
  }

  const uint32_t base_center_index = static_cast<uint32_t>(vertices.size());
  vertices.push_back(Vertex{
    .position = {0.0f, 0.0f, 0.0f},
    .normal = {0.0f, -1.0f, 0.0f},
    .uv = {0.5f, 0.5f},
  });

  for (uint32_t i = 0; i < segments; ++i) {
    float angle = circumference * (float(i) / float(segments));
    float x = radius * std::cos(angle);
    float z = radius * std::sin(angle);

    vertices.push_back(Vertex{
      .position = {x, 0.0f, z},
      .normal = {0.0f, -1.0f, 0.0f},
      .uv = {0.5f + (x / (2.0f * radius)), 0.5f + (z / (2.0f * radius))},
    });
  }

  const uint32_t tip_index = 0;
  const uint32_t side_start = 1;
  const uint32_t base_start = base_center_index + 1;


  for (uint32_t i = 0; i < segments; ++i) {
    uint32_t current = side_start + i;
    uint32_t next = side_start + ((i + 1) % segments);

    indices.push_back(tip_index);
    indices.push_back(current);
    indices.push_back(next);
  }

  for (uint32_t i = 0; i < segments; ++i) {
    uint32_t current = base_start + i;
    uint32_t next = base_start + ((i + 1) % segments);

    indices.push_back(base_center_index);
    indices.push_back(next);
    indices.push_back(current);
  }

  Mesh mesh;
  mesh.vertex_buffer = new veekay::graphics::Buffer(
    vertices.size() * sizeof(Vertex), vertices.data(),
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  mesh.index_buffer = new veekay::graphics::Buffer(
    indices.size() * sizeof(uint32_t), indices.data(),
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  mesh.indices = static_cast<uint32_t>(indices.size());

  return mesh;
}

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

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
  auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

  return view() * projection;
}

VkShaderModule loadShaderModule(const char* path) {
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

void initialize(VkCommandBuffer cmd) {
  VkDevice& device = veekay::app.vk_device;
  
  { 
    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    if (!vertex_shader_module) {
      std::cerr << "Failed to load Vulkan vertex shader from file\n";
      veekay::app.running = false;
      return;
    }

    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
    if (!fragment_shader_module) {
      std::cerr << "Failed to load Vulkan fragment shader from file\n";
      veekay::app.running = false;
      return;
    }

    VkPipelineShaderStageCreateInfo stage_infos[2];

    stage_infos[0] = VkPipelineShaderStageCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertex_shader_module,
      .pName = "main",
    };

    stage_infos[1] = VkPipelineShaderStageCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragment_shader_module,
      .pName = "main",
    };

    VkVertexInputBindingDescription buffer_binding{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription attributes[] = {
      {
        .location = 0, 
        .binding = 0, 
        .format = VK_FORMAT_R32G32B32_SFLOAT, 
        .offset = offsetof(Vertex, position), 
      },
      {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, normal),
      },
      {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(Vertex, uv),
      },
    };

    VkPipelineVertexInputStateCreateInfo input_state_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &buffer_binding,
      .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
      .pVertexAttributeDescriptions = attributes,
    };

    VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineRasterizationStateCreateInfo raster_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo sample_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = false,
      .minSampleShading = 1.0f,
    };

    VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(veekay::app.window_width),
      .height = static_cast<float>(veekay::app.window_height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
    };

    VkRect2D scissor{
      .offset = {0, 0},
      .extent = {veekay::app.window_width, veekay::app.window_height},
    };

    VkPipelineViewportStateCreateInfo viewport_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

      .viewportCount = 1,
      .pViewports = &viewport,

      .scissorCount = 1,
      .pScissors = &scissor,
    };

    VkPipelineDepthStencilStateCreateInfo depth_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = true,
      .depthWriteEnable = true,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    VkPipelineColorBlendAttachmentState attachment_info{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                        VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT |
                        VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo blend_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

      .logicOpEnable = false,
      .logicOp = VK_LOGIC_OP_COPY,

      .attachmentCount = 1,
      .pAttachments = &attachment_info
    };


    {
      VkDescriptorPoolSize pools[] = {
        {
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 8,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 8,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 8,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 8,
        }
      };
      
      VkDescriptorPoolCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
        .pPoolSizes = pools,
      };

      if (vkCreateDescriptorPool(device, &info, nullptr,
                                 &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan descriptor pool\n";
        veekay::app.running = false;
        return;
      }
    }

    {
      VkDescriptorSetLayoutBinding bindings[] = {
        {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 2,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
      };

      VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
        .pBindings = bindings,
      };

      if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                      &descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan descriptor set layout\n";
        veekay::app.running = false;
        return;
      }
    }

    {
      VkDescriptorSetAllocateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
      };

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

    if (vkCreatePipelineLayout(device, &layout_info,
                               nullptr, &pipeline_layout) != VK_SUCCESS) {
      std::cerr << "Failed to create Vulkan pipeline layout\n";
      veekay::app.running = false;
      return;
    }
    
    VkGraphicsPipelineCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stage_infos,
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

    if (vkCreateGraphicsPipelines(device, nullptr,
                                  1, &info, nullptr, &pipeline) != VK_SUCCESS) {
      std::cerr << "Failed to create Vulkan pipeline\n";
      veekay::app.running = false;
      return;
    }
  }

  scene_uniforms_buffer = new veekay::graphics::Buffer(
    sizeof(SceneUniforms),
    nullptr,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);


  model_uniforms_buffer = new veekay::graphics::Buffer(
    max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
    nullptr,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  point_lights_buffer = new veekay::graphics::Buffer(
    sizeof(PointLight) * max_point_lights,
    nullptr,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  {
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
        .buffer = point_lights_buffer->buffer,
        .offset = 0,
        .range = sizeof(PointLight) * 16,
      },
    };

    VkWriteDescriptorSet write_infos[] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_infos[0],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .pBufferInfo = &buffer_infos[1],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[2],
      },
    };

    vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                           write_infos, 0, nullptr);
  }

  {
    constexpr uint32_t cone_segments = 80;
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
      instance.transform.rotation = {};
      instance.albedo_color = cone_color;
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

  delete point_lights_buffer;
  delete model_uniforms_buffer;
  delete scene_uniforms_buffer;

  vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyShaderModule(device, fragment_shader_module, nullptr);
  vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
  ImGui::Begin("Controls:");
  ImGui::SliderFloat("Rotation speed", &rotation_speed_multiplier, 0.0f, 5.0f);
  ImGui::Text("Camera Pos: (%.2f, %.2f, %.2f)", camera.position.x, camera.position.y, camera.position.z);
  ImGui::End();

  // --- УПРАВЛЕНИЕ ---
  if (!ImGui::IsWindowHovered()) { 
    using namespace veekay::input;
    
    constexpr float move_speed = 0.1f;

    // Мышь
    if (mouse::isButtonDown(mouse::Button::left)) {
        constexpr float sensitivity = 0.003f;
        auto move_delta = mouse::cursorDelta();

        camera.rotation.y += move_delta[0] * sensitivity;
        camera.rotation.x += move_delta[1] * sensitivity;
    }

    // Клавиатура
    const veekay::vec3 front = camera.getFront();
    veekay::vec3 up = {0.0f, 1.0f, 0.0f};
    const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, up));
    // Корректируем UP для вертикального стрейфа
    up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

    if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * move_speed;

    if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::z)) camera.position += up * move_speed;
  }
  // ------------------

  float delta_time = 0.0f;
  if (has_previous_time) {
    delta_time = static_cast<float>(time - previous_time);
  } else {
    has_previous_time = true;
  }
  previous_time = time;

  delta_time = std::max(delta_time, 0.0f);

  const float scaled_delta = delta_time * rotation_speed_multiplier;
  for (Model& model : models) {
    const float angle_delta = model.rotation_speed * scaled_delta;
    model.transform.rotation += model.rotation_axis * angle_delta;
  }

  std::vector<PointLight> point_lights;
  
  const float orbit_radius = 3.0f;
  const float orbit_height = 1.5f;
  PointLight orbiting_light{
    .position = {
      orbit_radius * std::cos(static_cast<float>(time)),
      orbit_height,
      orbit_radius * std::sin(static_cast<float>(time))
    },
    .color = {1.0f, 0.8f, 0.6f},
  };
  point_lights.push_back(orbiting_light);

  PointLight stationary_light{
    .position = {0.0f, 2.0f, 0.0f},
    .color = {0.6f, 0.8f, 1.0f},
  };
  point_lights.push_back(stationary_light);

  float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
  veekay::mat4 view_matrix = camera.view();
  SceneUniforms scene_uniforms{
    .view_projection = camera.view_projection(aspect_ratio),
    .view_inverse = veekay::mat4::inverse(view_matrix),
    .camera_position = camera.position,
    .global_light = {
      .ambient_color = {0.2f, 0.2f, 0.2f},
      .directional_color = {0.8f, 0.8f, 0.8f},
      .directional_direction = veekay::vec3::normalized({-0.5f, -1.0f, -0.2f}),
    },
    .num_point_lights = static_cast<uint32_t>(point_lights.size()),
  };

  std::vector<ModelUniforms> model_uniforms(models.size());
  for (size_t i = 0, n = models.size(); i < n; ++i) {
    const Model& model = models[i];
    ModelUniforms& uniforms = model_uniforms[i];

    uniforms.model = model.transform.matrix();
    uniforms.normal_matrix = veekay::mat4::transpose(veekay::mat4::inverse(uniforms.model));
    uniforms.albedo_color = model.albedo_color;
    uniforms.shininess = 32.0f;
    uniforms.specular_color = {0.5f, 0.5f, 0.5f};
  }

  *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

  const size_t alignment =
    veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

  for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
    const ModelUniforms& uniforms = model_uniforms[i];

    char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
    *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
  }

  if (!point_lights.empty()) {
    std::memcpy(point_lights_buffer->mapped_region, point_lights.data(),
                sizeof(PointLight) * point_lights.size());
  }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
  vkResetCommandBuffer(cmd, 0);

  { 
    VkCommandBufferBeginInfo info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

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
      .renderArea = {
        .extent = {
          veekay::app.window_width,
          veekay::app.window_height
        },
      },
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

    if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
      current_vertex_buffer = mesh.vertex_buffer->buffer;
      vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
    }

    if (current_index_buffer != mesh.index_buffer->buffer) {
      current_index_buffer = mesh.index_buffer->buffer;
      vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
    }

    uint32_t offset = i * model_uniorms_alignment;
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
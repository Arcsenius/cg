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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_lights = 16; // Для point и spot lights

struct Vertex {
  veekay::vec3 position;
  veekay::vec3 normal;
  veekay::vec2 uv;
};

struct LightColors {
    veekay::vec3 ambient; float _pad0;
    veekay::vec3 diffuse; float _pad1;
    veekay::vec3 specular; float _pad2;
};

struct DirectionalLight {
    veekay::vec3 direction;
    float intensity;
    LightColors colors;
};

struct AmbientLight {
    veekay::vec3 color;
    float intensity;
    veekay::vec3 specular_color;
    float shininess;
};

struct PointLight {
    veekay::vec3 position = {};
    float _pad0;
    LightColors colors = {
        .ambient = {0.0f, 0.0f, 0.0f},
        .diffuse = {1.0f, 1.0f, 1.0f},
        .specular = {1.0f, 1.0f, 1.0f}
    };
    float linear = 0.09f;
    float quadratic = 0.032f;
    float _pad1;
    float _pad2;
};

struct SpotLight {
    PointLight point_light;
    veekay::vec3 direction;
    float cut_off;       // cos(angle)
    float outer_cut_off; // cos(outer_angle)
    float _pad1;
    float _pad2;
};
struct alignas(16) SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 camera_position;
    uint32_t num_point_lights = 0;
    uint32_t num_spot_lights = 0;
    uint32_t _pad_align[3]; // Выравнивание
    DirectionalLight directional_light;
    AmbientLight ambient_light;
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

struct Camera {
  constexpr static float default_fov = 70.0f;
  constexpr static float default_near_plane = 0.01f;
  constexpr static float default_far_plane = 100.0f;

  veekay::vec3 position = {};
  veekay::vec3 rotation = {};

  float fov = default_fov;
  float near_plane = default_near_plane;
  float far_plane = default_far_plane;

  static float toRadians(float degrees) { return degrees * static_cast<float>(M_PI) / 180.0f; }

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

  veekay::mat4 view() const {
      auto rotation_matrix = [](const veekay::vec3& axis, float angle) {
          if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) return veekay::mat4::identity();
          return veekay::mat4::rotation(axis, angle);
      };
      auto translation = veekay::mat4::translation(position);
      auto rotate_x = rotation_matrix({1.0f, 0.0f, 0.0f}, rotation.x);
      auto rotate_y = rotation_matrix({0.0f, 1.0f, 0.0f}, rotation.y);
      veekay::mat4 camera_world_matrix = translation * rotate_y * rotate_x;
      return veekay::mat4::inverse(camera_world_matrix);
  }

  veekay::mat4 view_projection(float aspect_ratio) const;
};
inline namespace {
  Camera camera{
    .position = {0.0f, -0.5f, -3.0f}
  };

  std::vector<Model> models;
  DirectionalLight dir_light{
      .direction = {0.3f, -1.0f, 0.5f},
      .intensity = 0.8f,
      .colors = {
          .ambient = {0.2f, 0.2f, 0.2f},
          .diffuse = {1.0f, 1.0f, 0.95f},
          .specular = {1.0f, 1.0f, 1.0f}
      }
  };

  AmbientLight ambient_light{
      .color = {1.0f, 1.0f, 1.0f},
      .intensity = 0.05f,
      .specular_color = {0.1f, 0.1f, 0.1f},
      .shininess = 32.0f,
  };

  std::vector<PointLight> point_lights;
  std::vector<SpotLight> spot_lights;
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
  veekay::graphics::Buffer* spot_lights_buffer; // <--- НОВЫЙ БУФЕР

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
  vertices.push_back(Vertex{{0.0f, height, 0.0f},{0.0f, 1.0f, 0.0f},{0.5f, 0.0f}});
  const float circumference = 2.0f * float(M_PI);
  for (uint32_t i = 0; i < segments; ++i) {
    float angle = circumference * (float(i) / float(segments));
    float x = radius * std::cos(angle);
    float z = radius * std::sin(angle);
    veekay::vec3 normal = veekay::vec3::normalized({x, radius / height, z});
    vertices.push_back(Vertex{{x, 0.0f, z}, normal, {float(i) / float(segments), 1.0f}});
  }
  const uint32_t base_center_index = static_cast<uint32_t>(vertices.size());
  vertices.push_back(Vertex{{0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});
  for (uint32_t i = 0; i < segments; ++i) {
    float angle = circumference * (float(i) / float(segments));
    float x = radius * std::cos(angle);
    float z = radius * std::sin(angle);
    vertices.push_back(Vertex{{x, 0.0f, z}, {0.0f, -1.0f, 0.0f}, {0.5f + (x / (2.0f * radius)), 0.5f + (z / (2.0f * radius))}});
  }
  const uint32_t tip_index = 0;
  const uint32_t side_start = 1;
  const uint32_t base_start = base_center_index + 1;
  for (uint32_t i = 0; i < segments; ++i) {
    uint32_t current = side_start + i;
    uint32_t next = side_start + ((i + 1) % segments);
    indices.push_back(tip_index); indices.push_back(current); indices.push_back(next);
  }
  for (uint32_t i = 0; i < segments; ++i) {
    uint32_t current = base_start + i;
    uint32_t next = base_start + ((i + 1) % segments);
    indices.push_back(base_center_index); indices.push_back(next); indices.push_back(current);
  }
  Mesh mesh;
  mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  mesh.indices = static_cast<uint32_t>(indices.size());
  return mesh;
}

veekay::mat4 Transform::matrix() const {
  auto rotation_matrix = [](const veekay::vec3& axis, float angle) {
    if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) return veekay::mat4::identity();
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
  VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = buffer.data()};
  VkShaderModule result;
  if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) return nullptr;
  return result;
}

void initialize(VkCommandBuffer cmd) {
  VkDevice& device = veekay::app.vk_device;
  
  { 
    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
    if (!vertex_shader_module || !fragment_shader_module) {
      std::cerr << "Failed to load shaders\n";
      veekay::app.running = false;
      return;
    }

    VkPipelineShaderStageCreateInfo stage_infos[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader_module, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader_module, .pName = "main"}
    };

    VkVertexInputBindingDescription buffer_binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
    };
    VkPipelineVertexInputStateCreateInfo input_state_info{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &buffer_binding, 3, attributes};
    VkPipelineInputAssemblyStateCreateInfo assembly_state_info{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false};
    VkPipelineRasterizationStateCreateInfo raster_info{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, false, false, VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, false, 0.0f, 0.0f, 0.0f, 1.0f};
    VkPipelineMultisampleStateCreateInfo sample_info{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, false, 1.0f, nullptr, false, false};
    VkViewport viewport{0.0f, 0.0f, (float)veekay::app.window_width, (float)veekay::app.window_height, 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {veekay::app.window_width, veekay::app.window_height}};
    VkPipelineViewportStateCreateInfo viewport_info{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &viewport, 1, &scissor};
    VkPipelineDepthStencilStateCreateInfo depth_info{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, false, false, {}, {}};
    VkPipelineColorBlendAttachmentState attachment_info{false, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF};
    VkPipelineColorBlendStateCreateInfo blend_info{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, false, VK_LOGIC_OP_COPY, 1, &attachment_info, {0.0f,0.0f,0.0f,0.0f}};
    {
      VkDescriptorPoolSize pools[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}, // Увеличили кол-во storage buffers
      };
      VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 2, 3, pools};
      vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
    }
    {
      VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // Point Lights
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // Spot Lights (NEW)
      };
      VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings};
      vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout);
    }
    {
      VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptor_pool, 1, &descriptor_set_layout};
      vkAllocateDescriptorSets(device, &info, &descriptor_set);
    }

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descriptor_set_layout, 0, nullptr};
    vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
    
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stage_infos, &input_state_info, &assembly_state_info, nullptr, &viewport_info, &raster_info, &sample_info, &depth_info, &blend_info, nullptr, pipeline_layout, veekay::app.vk_render_pass, 0, VK_NULL_HANDLE, 0};
    vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline);
  }
  scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  model_uniforms_buffer = new veekay::graphics::Buffer(max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  point_lights_buffer = new veekay::graphics::Buffer(sizeof(PointLight) * max_lights, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  spot_lights_buffer = new veekay::graphics::Buffer(sizeof(SpotLight) * max_lights, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); // NEW
  {
    VkDescriptorBufferInfo buffer_infos[] = {
      {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
      {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
      {point_lights_buffer->buffer, 0, sizeof(PointLight) * max_lights},
      {spot_lights_buffer->buffer, 0, sizeof(SpotLight) * max_lights}, // NEW
    };
    VkWriteDescriptorSet write_infos[] = {
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[3], nullptr}, // NEW
    };
    vkUpdateDescriptorSets(device, 4, write_infos, 0, nullptr);
  }
  {
    constexpr uint32_t cone_segments = 80;
    cone_mesh = createConeMesh(cone_segments, 0.6f, 1.4f);
  }
  {
    struct ConeConfig { veekay::vec3 pos; veekay::vec3 scale; veekay::vec3 axis; float speed; };
    const std::array<ConeConfig, 5> cones = {{
      {{-2.3f, -0.7f, -2.2f}, {0.9f, 0.9f, 0.9f}, {0.0f, 1.0f, 0.0f}, 0.6f},
      {{-0.8f, -0.7f, -1.0f}, {0.7f, 1.1f, 0.7f}, {0.3f, 1.0f, 0.1f}, 0.9f},
      {{1.6f, -0.7f, -1.4f}, {1.0f, 0.8f, 1.0f}, {0.0f, 1.0f, 0.4f}, 1.3f},
      {{0.4f, -0.7f, -2.6f}, {0.6f, 1.2f, 0.6f}, {0.2f, 1.0f, 0.3f}, 0.8f},
      {{2.2f, -0.7f, -0.8f}, {1.1f, 0.9f, 0.9f}, {0.4f, 1.0f, 0.0f}, 1.6f},
    }};

    models.clear();
    for (const auto& cone : cones) {
      Model instance;
      instance.mesh = cone_mesh;
      instance.transform.position = cone.pos;
      instance.transform.scale = cone.scale;
      instance.transform.rotation = {};
      instance.albedo_color = cone_color;
      instance.rotation_axis = veekay::vec3::normalized(cone.axis);
      instance.rotation_speed = cone.speed;
      models.emplace_back(instance);
    }
  }
  point_lights.push_back({
      .position = {0.0f, 2.0f, 0.0f},
      .colors = {.ambient={0.1f,0.1f,0.1f}, .diffuse={0.6f,0.8f,1.0f}, .specular={1.0f,1.0f,1.0f}},
      .linear = 0.09f, .quadratic = 0.032f
  });

  spot_lights.push_back({
      .point_light = {
          .position = {0.0f, 0.0f, 2.0f},
          .colors = {.ambient={0.0f,0.0f,0.0f}, .diffuse={1.0f,0.0f,0.0f}, .specular={1.0f,1.0f,1.0f}},
          .linear = 0.09f, .quadratic = 0.032f
      },
      .direction = {0.0f, 0.0f, -1.0f},
      .cut_off = std::cos(Camera::toRadians(12.5f)),
      .outer_cut_off = std::cos(Camera::toRadians(17.5f))
  });
}

void shutdown() {
  VkDevice& device = veekay::app.vk_device;
  delete cone_mesh.index_buffer; delete cone_mesh.vertex_buffer;
  delete spot_lights_buffer; delete point_lights_buffer;
  delete model_uniforms_buffer; delete scene_uniforms_buffer;
  vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyShaderModule(device, fragment_shader_module, nullptr);
  vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
  ImGui::Begin("Controls");
  ImGui::SliderFloat("Rotation speed", &rotation_speed_multiplier, 0.0f, 5.0f);
  if (ImGui::CollapsingHeader("Directional Light")) {
      ImGui::PushID("dir_light"); // <--- НАЧАЛО УНИКАЛЬНОЙ ЗОНЫ
      ImGui::SliderFloat3("Direction", &dir_light.direction.x, -1.0f, 1.0f);
      ImGui::SliderFloat("Intensity", &dir_light.intensity, 0.0f, 5.0f);
      ImGui::ColorEdit3("Ambient", &dir_light.colors.ambient.x);
      ImGui::ColorEdit3("Diffuse", &dir_light.colors.diffuse.x);
      ImGui::ColorEdit3("Specular", &dir_light.colors.specular.x);
      ImGui::PopID(); // <--- КОНЕЦ УНИКАЛЬНОЙ ЗОНЫ
  }
  if (ImGui::CollapsingHeader("Ambient Light")) {
      ImGui::PushID("amb_light"); // Уникальный ID для этого блока
      ImGui::ColorEdit3("Color", &ambient_light.color.x);
      ImGui::SliderFloat("Intensity", &ambient_light.intensity, 0.0f, 1.0f);
      ImGui::PopID();
  }
  if (ImGui::CollapsingHeader("Point Light 0")) {
      ImGui::PushID("point_light_0"); // Теперь "Diffuse" внутри этого блока не конфликтует с другими
      PointLight& pl = point_lights[0];
      ImGui::SliderFloat3("Position", &pl.position.x, -5.0f, 5.0f);
      ImGui::ColorEdit3("Diffuse", &pl.colors.diffuse.x);
      ImGui::SliderFloat("Linear", &pl.linear, 0.001f, 1.0f);
      ImGui::SliderFloat("Quadratic", &pl.quadratic, 0.001f, 1.0f);
      ImGui::PopID();
  }
  if (ImGui::CollapsingHeader("Spot Light 0")) {
      ImGui::PushID("spot_light_0"); // Уникальный ID
      SpotLight& sl = spot_lights[0];
      ImGui::SliderFloat3("Position", &sl.point_light.position.x, -5.0f, 5.0f);
      ImGui::SliderFloat3("Direction", &sl.direction.x, -1.0f, 1.0f);
      ImGui::ColorEdit3("Diffuse", &sl.point_light.colors.diffuse.x);
      
      float cutoff_deg = std::acos(sl.cut_off) * 180.0f / (float)M_PI;
      float outer_deg = std::acos(sl.outer_cut_off) * 180.0f / (float)M_PI;
      // Но внутри PushID это происходит автоматически для внутренних ID
      ImGui::SliderFloat("Cutoff", &cutoff_deg, 0.0f, 45.0f);
      ImGui::SliderFloat("Outer", &outer_deg, 0.0f, 45.0f);
      
      if (outer_deg < cutoff_deg) outer_deg = cutoff_deg; // Защита от инверсии

      sl.cut_off = std::cos(Camera::toRadians(cutoff_deg));
      sl.outer_cut_off = std::cos(Camera::toRadians(outer_deg));
      ImGui::PopID();
  }
  ImGui::End();

  if (!ImGui::IsWindowHovered()) { 
    using namespace veekay::input;
    constexpr float move_speed = 0.1f;
    if (mouse::isButtonDown(mouse::Button::left)) {
        auto move_delta = mouse::cursorDelta();
        camera.rotation.y += move_delta[0] * 0.003f;
        camera.rotation.x += move_delta[1] * 0.003f;
    }
    const veekay::vec3 front = camera.getFront();
    veekay::vec3 up = {0.0f, 1.0f, 0.0f};
    const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, up));
    up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

    if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * move_speed;
    if (keyboard::isKeyDown(keyboard::Key::z)) camera.position += up * move_speed;
  }
  float delta_time = 0.0f;
  if (has_previous_time) delta_time = std::max((float)(time - previous_time), 0.0f);
  else has_previous_time = true;
  previous_time = time;
  for (Model& model : models) {
    model.transform.rotation += model.rotation_axis * (model.rotation_speed * delta_time * rotation_speed_multiplier);
  }
  float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
  SceneUniforms scene_uniforms{
    .view_projection = camera.view_projection(aspect_ratio),
    .camera_position = camera.position,
    .num_point_lights = (uint32_t)point_lights.size(),
    .num_spot_lights = (uint32_t)spot_lights.size(),
    .directional_light = dir_light,
    .ambient_light = ambient_light
  };
  *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
  std::vector<ModelUniforms> model_uniforms(models.size());
  for (size_t i = 0; i < models.size(); ++i) {
    model_uniforms[i] = {
        .model = models[i].transform.matrix(),
        .normal_matrix = veekay::mat4::transpose(veekay::mat4::inverse(models[i].transform.matrix())),
        .albedo_color = models[i].albedo_color,
        .shininess = 32.0f,
        .specular_color = {0.5f, 0.5f, 0.5f}
    };
  }
  const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
  for (size_t i = 0; i < model_uniforms.size(); ++i) {
    *(ModelUniforms*)((char*)model_uniforms_buffer->mapped_region + i * alignment) = model_uniforms[i];
  }
  if (!point_lights.empty())
    std::memcpy(point_lights_buffer->mapped_region, point_lights.data(), sizeof(PointLight) * point_lights.size());
  if (!spot_lights.empty())
    std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), sizeof(SpotLight) * spot_lights.size());
}
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cmd, &info);

  VkClearValue clear_values[] = {{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}}, {.depthStencil = {1.0f, 0}}};
  VkRenderPassBeginInfo rp_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, veekay::app.vk_render_pass, framebuffer, {{0,0}, {veekay::app.window_width, veekay::app.window_height}}, 2, clear_values};
  vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  VkDeviceSize zero_offset = 0;
  VkBuffer current_vertex = VK_NULL_HANDLE, current_index = VK_NULL_HANDLE;
  const size_t align = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

  for (size_t i = 0; i < models.size(); ++i) {
    const Model& model = models[i];
    if (current_vertex != model.mesh.vertex_buffer->buffer) {
      current_vertex = model.mesh.vertex_buffer->buffer;
      vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex, &zero_offset);
    }
    if (current_index != model.mesh.index_buffer->buffer) {
      current_index = model.mesh.index_buffer->buffer;
      vkCmdBindIndexBuffer(cmd, current_index, zero_offset, VK_INDEX_TYPE_UINT32);
    }
    uint32_t offset = i * align;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
    vkCmdDrawIndexed(cmd, model.mesh.indices, 1, 0, 0, 0);
  }
  vkCmdEndRenderPass(cmd);
  vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
  return veekay::run({.init = initialize, .shutdown = shutdown, .update = update, .render = render});
}
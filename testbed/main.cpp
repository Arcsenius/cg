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
#include <stdio.h>
#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_lights = 16; 

struct Vertex {
  veekay::vec3 position;
  veekay::vec3 normal;
  veekay::vec2 uv;
};

// --- СТРУКТУРЫ СВЕТА ---

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
    float cut_off;       
    float outer_cut_off; 
    float _pad1;
    float _pad2;
};

// ВАЖНОЕ ИСПРАВЛЕНИЕ: Используем vec4 для camera_position и явный паддинг
// Это гарантирует совпадение с GLSL std140
struct alignas(16) SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec4 camera_position; // xyz = pos, w = unused
    uint32_t num_point_lights = 0;
    uint32_t num_spot_lights = 0;
    float _pad_align[2]; 
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

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
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
    .position = {0.0f, 0.0f, -4.0f}
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
      .intensity = 0.3f,
      .specular_color = {0.1f, 0.1f, 0.1f},
      .shininess = 32.0f,
  };

  std::vector<PointLight> point_lights;
  std::vector<SpotLight> spot_lights;

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
  veekay::graphics::Buffer* spot_lights_buffer;

  Mesh cube_mesh;
  Texture tex_side;
  Texture tex_top;
  Texture tex_bottom;

  float rotation_speed_multiplier = 1.0f;
  double previous_time = 0.0;
  bool has_previous_time = false;
}

// --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

uint32_t getGraphicsQueueFamily() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(veekay::app.vk_physical_device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(veekay::app.vk_physical_device, &queueFamilyCount, queueFamilies.data());
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    }
    throw std::runtime_error("failed to find graphics queue family!");
}

Texture loadTexture(const char* filename) {
    Texture tex;
    std::vector<unsigned char> image;
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, filename);
    
    // ВАЖНО: Проверка ошибки
    if(error) {
        std::cerr << "!!! ERROR LOADING TEXTURE: " << filename << "\n"
                  << "Error code: " << error << ": " << lodepng_error_text(error) << "\n"
                  << "Make sure the .png file is in the same folder as the executable!" << std::endl;
        // Возвращаем пустую текстуру (потом это может вызвать краш, но мы увидим ошибку в консоли)
        return tex;
    } else {
        std::cout << "Successfully loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
    }

    VkDevice device = veekay::app.vk_device;
    VkDeviceSize imageSize = image.size();
    
    uint32_t queueFamilyIndex = getGraphicsQueueFamily();
    VkQueue graphicsQueue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &graphicsQueue);

    VkCommandPool commandPool;
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory);
    vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, image.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vkCreateImage(device, &imageInfo, nullptr, &tex.image);

    vkGetImageMemoryRequirements(device, tex.image, &memRequirements);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &tex.memory);
    vkBindImageMemory(device, tex.image, tex.memory, 0);

    VkCommandBufferAllocateInfo allocCmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandPool = commandPool;
    allocCmdInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocCmdInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &viewInfo, nullptr, &tex.view);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkCreateSampler(device, &samplerInfo, nullptr, &tex.sampler);

    return tex;
}

Mesh createCubeMesh() {
    std::vector<Vertex> vertices = {
        // Front
        {{-0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 0.0f}},
        {{-0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 0.0f}},
        // Back
        {{ 0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 1.0f}},
        {{-0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 0.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 0.0f}},
        // Left
        {{-0.5f, -0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
        {{-0.5f,  0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
        // Right
        {{ 0.5f, -0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
        // Top
        {{-0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 0.0f}},
        {{-0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 0.0f}},
        // Bottom
        {{-0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 0.0f}},
        {{-0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 0.0f}},
    };

    std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0,       
        4, 5, 6, 6, 7, 4,       
        8, 9, 10, 10, 11, 8,    
        12, 13, 14, 14, 15, 12, 
        16, 17, 18, 18, 19, 16, 
        20, 21, 22, 22, 23, 20  
    };

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
  if(!file.is_open()) {
      std::cerr << "!!! ERROR: Failed to open shader file: " << path << std::endl;
      return nullptr;
  }
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
  
  // ЗАГРУЗКА ТЕКСТУР
  tex_side = loadTexture("tnt_side.png");
  tex_top = loadTexture("tnt_top.png");
  tex_bottom = loadTexture("tnt_bottom.png");

  { 
    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
    if (!vertex_shader_module || !fragment_shader_module) {
      std::cerr << "Failed to load shaders! Did you compile them?\n";
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

    // --- ОБНОВЛЕННЫЕ ДЕСКРИПТОРЫ (7 ШТУК) ---
    {
      VkDescriptorPoolSize pools[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16} 
      };
      VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 4, 4, pools};
      vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
    }
    {
      VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // Текстуры
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // Side
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // Top
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // Bottom
      };
      VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, bindings};
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
  spot_lights_buffer = new veekay::graphics::Buffer(sizeof(SpotLight) * max_lights, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  {
    VkDescriptorBufferInfo buffer_infos[] = {
      {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
      {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
      {point_lights_buffer->buffer, 0, sizeof(PointLight) * max_lights},
      {spot_lights_buffer->buffer, 0, sizeof(SpotLight) * max_lights},
    };
    
    // Если текстура не загрузилась (null), использование ее в дескрипторе недопустимо,
    // но чтобы программа не падала сразу, мы надеемся на лучшее (или надо делать заглушку)
    // В данном примере полагаемся на то, что файлы на месте.
    VkDescriptorImageInfo image_infos[] = {
        {tex_side.sampler, tex_side.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {tex_top.sampler, tex_top.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {tex_bottom.sampler, tex_bottom.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    };

    VkWriteDescriptorSet write_infos[] = {
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[3], nullptr},
      // Текстуры
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[0], nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[1], nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[2], nullptr, nullptr},
    };
    vkUpdateDescriptorSets(device, 7, write_infos, 0, nullptr);
  }

  cube_mesh = createCubeMesh();

  {
    struct CubeConfig { veekay::vec3 pos; veekay::vec3 scale; veekay::vec3 axis; float speed; };
    const std::array<CubeConfig, 3> cubes = {{
      {{0.0f, 0.0f, -2.0f}, {1.0f, 1.0f, 1.0f}, {0.3f, 1.0f, 0.1f}, 0.5f},
      {{-1.5f, 0.5f, -3.0f}, {0.8f, 0.8f, 0.8f}, {1.0f, 0.2f, 0.0f}, 1.0f},
      {{1.5f, -0.5f, -2.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, 0.2f},
    }};

    models.clear();
    for (const auto& cube : cubes) {
      Model instance;
      instance.mesh = cube_mesh;
      instance.transform.position = cube.pos;
      instance.transform.scale = cube.scale;
      instance.transform.rotation = {};
      instance.albedo_color = {1.0f, 1.0f, 1.0f}; 
      instance.rotation_axis = veekay::vec3::normalized(cube.axis);
      instance.rotation_speed = cube.speed;
      models.emplace_back(instance);
    }
  }

  point_lights.push_back({
      .position = {0.0f, 2.0f, 0.0f},
      .colors = {.ambient={0.1f,0.1f,0.1f}, .diffuse={0.8f,0.8f,0.8f}, .specular={1.0f,1.0f,1.0f}},
  });
}

void shutdown() {
  VkDevice& device = veekay::app.vk_device;
  delete cube_mesh.index_buffer; delete cube_mesh.vertex_buffer;
  delete spot_lights_buffer; delete point_lights_buffer;
  delete model_uniforms_buffer; delete scene_uniforms_buffer;
  
  vkDestroySampler(device, tex_side.sampler, nullptr); vkDestroyImageView(device, tex_side.view, nullptr); vkDestroyImage(device, tex_side.image, nullptr); vkFreeMemory(device, tex_side.memory, nullptr);
  vkDestroySampler(device, tex_top.sampler, nullptr); vkDestroyImageView(device, tex_top.view, nullptr); vkDestroyImage(device, tex_top.image, nullptr); vkFreeMemory(device, tex_top.memory, nullptr);
  vkDestroySampler(device, tex_bottom.sampler, nullptr); vkDestroyImageView(device, tex_bottom.view, nullptr); vkDestroyImage(device, tex_bottom.image, nullptr); vkFreeMemory(device, tex_bottom.memory, nullptr);

  vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyShaderModule(device, fragment_shader_module, nullptr);
  vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
  // UI и управление
  ImGui::Begin("Controls");
  ImGui::SliderFloat("Rotation speed", &rotation_speed_multiplier, 0.0f, 5.0f);
  if (ImGui::CollapsingHeader("Directional Light")) {
      ImGui::SliderFloat3("Dir", &dir_light.direction.x, -1.0f, 1.0f);
      ImGui::SliderFloat("Int", &dir_light.intensity, 0.0f, 5.0f);
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
    .camera_position = {camera.position.x, camera.position.y, camera.position.z, 0.0f}, // vec4
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
#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct DirectionalLight {
    veekay::vec3 direction;
    float _pad0;
    veekay::vec3 ambient;
    float _pad1;
    veekay::vec3 diffuse;
    float _pad2;
    veekay::vec3 specular;
    float _pad3;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 view_position;
    float _pad0;
    DirectionalLight directional_light;
    uint32_t point_light_count;
    uint32_t spot_light_count;
    float _pad1[2];
    veekay::mat4 light_space_matrix;
};

struct ModelUniforms {
    veekay::mat4 model;
    veekay::vec3 albedo_color;
    float shininess;
    veekay::vec3 specular_color;
    float _pad0;
};

struct Material {
    veekay::vec3 albedo;
    veekay::vec3 specular;
    float shininess;
    
    veekay::graphics::Texture* texture;
    VkSampler sampler;
    VkDescriptorSet descriptor_set;
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
    Material material;
    // Дополнительные параметры для вращения
    veekay::vec3 rotation_axis = {0.0f, 1.0f, 0.0f}; 
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    veekay::vec3 position = {};
    veekay::vec3 rotation = {};

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    veekay::mat4 view() const;
    veekay::mat4 view_projection(float aspect_ratio) const;
};

inline namespace {
    Camera camera{
        .position = {0.0f, -4.0f, -9.0f},
        .rotation = {-30.0f, 0.0f, 0.0f}
    };

    std::vector<Model> models;
}

inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;

    Mesh plane_mesh;
    Mesh cone_mesh;
    Mesh cube_mesh; // Меш куба

    veekay::graphics::Texture* missing_texture;
    VkSampler missing_texture_sampler;
    
    veekay::graphics::Texture* floor_texture;
    VkSampler floor_sampler;

    veekay::graphics::Texture* cone_texture;
    VkSampler cone_sampler;

    veekay::graphics::Texture* cube_texture; // Текстура куба
    VkSampler cube_sampler;

    constexpr uint32_t shadow_map_size = 2048;

    VkImage shadow_map_image;
    VkImageView shadow_map_view;
    VkDeviceMemory shadow_map_memory;

    VkSampler shadow_map_sampler;

    VkShaderModule shadow_vertex_shader_module;
    VkShaderModule shadow_fragment_shader_module;

    VkPipelineLayout shadow_pipeline_layout;
    VkPipeline shadow_pipeline;

    VkDescriptorSetLayout shadow_descriptor_set_layout;
    VkDescriptorSet shadow_descriptor_set;

    VkRenderPass shadow_render_pass;
    VkFramebuffer shadow_framebuffer;

    veekay::graphics::Buffer* light_space_buffer;
    
    constexpr uint32_t max_descriptor_sets = 32;
}

// --- ФУНКЦИЯ СОЗДАНИЯ КОНУСА (Исправленная: центр в середине) ---
Mesh createConeMesh(uint32_t segments, float radius, float height) {
    segments = std::max(segments, 3u);
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(1 + segments + 1 + segments);
    indices.reserve(segments * 6);
    
    float y_offset = -height / 2.0f;
    float slope = radius / height;

    // Вершина
    vertices.push_back(Vertex{{0.0f, height + y_offset, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.0f}});
    
    const float circumference = 2.0f * float(M_PI);
    
    // Бока
    for (uint32_t i = 0; i < segments; ++i) {
        float angle = circumference * (float(i) / float(segments));
        float nx = std::cos(angle);
        float nz = std::sin(angle);
        float x = radius * nx;
        float z = radius * nz;
        veekay::vec3 normal = veekay::vec3::normalized({nx, slope, nz});
        vertices.push_back(Vertex{{x, 0.0f + y_offset, z}, normal, {float(i) / float(segments), 1.0f}});
    }
    
    // Основание
    const uint32_t base_center_index = static_cast<uint32_t>(vertices.size());
    vertices.push_back(Vertex{{0.0f, 0.0f + y_offset, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});
    
    for (uint32_t i = 0; i < segments; ++i) {
        float angle = circumference * (float(i) / float(segments));
        float x = radius * std::cos(angle);
        float z = radius * std::sin(angle);
        vertices.push_back(Vertex{{x, 0.0f + y_offset, z}, {0.0f, -1.0f, 0.0f}, {0.5f + (x / (2.0f * radius)), 0.5f + (z / (2.0f * radius))}});
    }
    
    const uint32_t tip_index = 0;
    const uint32_t side_start = 1;
    const uint32_t base_start = base_center_index + 1;
    
    for (uint32_t i = 0; i < segments; ++i) {
        uint32_t current = side_start + i;
        uint32_t next = side_start + ((i + 1) % segments);
        indices.push_back(tip_index); indices.push_back(next); indices.push_back(current);
    }
    for (uint32_t i = 0; i < segments; ++i) {
        uint32_t current = base_start + i;
        uint32_t next = base_start + ((i + 1) % segments);
        indices.push_back(base_center_index); indices.push_back(current); indices.push_back(next);
    }
    
    Mesh mesh;
    mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.indices = static_cast<uint32_t>(indices.size());
    return mesh;
}

// --- ФУНКЦИЯ СОЗДАНИЯ КУБА ---
Mesh createCubeMesh(float size) {
    float h = size * 0.5f;

    std::vector<Vertex> vertices = {
        // Front
        {{-h,  h,  h}, { 0.0f,  0.0f,  1.0f}, {0.0f, 0.0f}},
        {{ h,  h,  h}, { 0.0f,  0.0f,  1.0f}, {1.0f, 0.0f}},
        {{ h, -h,  h}, { 0.0f,  0.0f,  1.0f}, {1.0f, 1.0f}},
        {{-h, -h,  h}, { 0.0f,  0.0f,  1.0f}, {0.0f, 1.0f}},
        // Back
        {{ h,  h, -h}, { 0.0f,  0.0f, -1.0f}, {0.0f, 0.0f}},
        {{-h,  h, -h}, { 0.0f,  0.0f, -1.0f}, {1.0f, 0.0f}},
        {{-h, -h, -h}, { 0.0f,  0.0f, -1.0f}, {1.0f, 1.0f}},
        {{ h, -h, -h}, { 0.0f,  0.0f, -1.0f}, {0.0f, 1.0f}},
        // Left
        {{-h,  h, -h}, {-1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
        {{-h,  h,  h}, {-1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
        {{-h, -h,  h}, {-1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
        {{-h, -h, -h}, {-1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
        // Right
        {{ h,  h,  h}, { 1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
        {{ h,  h, -h}, { 1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
        {{ h, -h, -h}, { 1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
        {{ h, -h,  h}, { 1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
        // Top
        {{-h,  h, -h}, { 0.0f,  1.0f,  0.0f}, {0.0f, 0.0f}},
        {{ h,  h, -h}, { 0.0f,  1.0f,  0.0f}, {1.0f, 0.0f}},
        {{ h,  h,  h}, { 0.0f,  1.0f,  0.0f}, {1.0f, 1.0f}},
        {{-h,  h,  h}, { 0.0f,  1.0f,  0.0f}, {0.0f, 1.0f}},
        // Bottom
        {{-h, -h,  h}, { 0.0f, -1.0f,  0.0f}, {0.0f, 0.0f}},
        {{ h, -h,  h}, { 0.0f, -1.0f,  0.0f}, {1.0f, 0.0f}},
        {{ h, -h, -h}, { 0.0f, -1.0f,  0.0f}, {1.0f, 1.0f}},
        {{-h, -h, -h}, { 0.0f, -1.0f,  0.0f}, {0.0f, 1.0f}},
    };

    std::vector<uint32_t> indices = {
        0,  1,  2,  2,  3,  0,  // Front
        4,  5,  6,  6,  7,  4,  // Back
        8,  9, 10, 10, 11,  8,  // Left
        12, 13, 14, 14, 15, 12, // Right
        16, 17, 18, 18, 19, 16, // Top
        20, 21, 22, 22, 23, 20  // Bottom
    };

    Mesh mesh;
    mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.indices = static_cast<uint32_t>(indices.size());
    return mesh;
}
// ----------------------------------------------------

float toRadians(float degrees) {
    return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
    veekay::mat4 s = veekay::mat4::identity();
    s.elements[0][0] = scale.x;
    s.elements[1][1] = scale.y;
    s.elements[2][2] = scale.z;
    
    float rx = toRadians(rotation.x);
    float ry = toRadians(rotation.y);
    float rz = toRadians(rotation.z);
    
    veekay::mat4 rot_x = veekay::mat4::identity();
    rot_x.elements[1][1] = cosf(rx);
    rot_x.elements[1][2] = -sinf(rx);
    rot_x.elements[2][1] = sinf(rx);
    rot_x.elements[2][2] = cosf(rx);
    
    veekay::mat4 rot_y = veekay::mat4::identity();
    rot_y.elements[0][0] = cosf(ry);
    rot_y.elements[0][2] = sinf(ry);
    rot_y.elements[2][0] = -sinf(ry);
    rot_y.elements[2][2] = cosf(ry);
    
    veekay::mat4 rot_z = veekay::mat4::identity();
    rot_z.elements[0][0] = cosf(rz);
    rot_z.elements[0][1] = -sinf(rz);
    rot_z.elements[1][0] = sinf(rz);
    rot_z.elements[1][1] = cosf(rz);
    
    veekay::mat4 t = veekay::mat4::translation(position);
    
    // ИСПРАВЛЕНИЕ: Сначала Scale, потом Rotate, потом Translate
    return t * (rot_z * (rot_y * (rot_x * s)));
}

veekay::mat4 Camera::view() const {
    auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(-rotation.x));
    auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(-rotation.y));
    auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(-rotation.z));
    auto r = ry * rx * rz;

    auto t = veekay::mat4::translation(-position);

    return t * r;
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

veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const char* path) {
    std::vector<unsigned char> image;
    unsigned width, height;
    
    unsigned error = lodepng::decode(image, width, height, path);
    
    if (error) {
        std::cerr << "Failed to load texture from " << path << ": " 
                  << lodepng_error_text(error) << "\n";
        return nullptr;
    }
    
    std::cout << "Loaded texture: " << path << " (" << width << "x" << height << ")\n";
    
    return new veekay::graphics::Texture(
        cmd, 
        width, 
        height,
        VK_FORMAT_R8G8B8A8_UNORM,
        image.data()
    );
}

VkSampler createSampler(VkFilter filter = VK_FILTER_LINEAR, 
                       VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT) {
    VkDevice& device = veekay::app.vk_device;
    
    VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = addressMode,
        .addressModeV = addressMode,
        .addressModeW = addressMode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16.0f,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    
    VkSampler sampler;
    if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan sampler\n";
        return VK_NULL_HANDLE;
    }
    
    return sampler;
}

veekay::mat4 calculateLightSpaceMatrix(const veekay::vec3& light_position) {
    float left = -15.0f;
    float right = 15.0f;
    float bottom = -15.0f;
    float top = 15.0f;
    float near_plane = 0.1f;
    float far_plane = 30.0f;
    
    veekay::vec3 eye = light_position;
    veekay::vec3 center = {0.0f, 0.0f, 0.0f};
    veekay::vec3 up = {0.0f, 0.0f, 1.0f};

    veekay::mat4 view = veekay::mat4::lookAt(eye, center, up);
    
    veekay::mat4 proj = veekay::mat4::identity();
    proj.elements[0][0] = 2.0f / (right - left);
    proj.elements[1][1] = 2.0f / (top - bottom);
    proj.elements[2][2] = -1.0f / (far_plane - near_plane);
    proj.elements[3][0] = -(right + left) / (right - left);
    proj.elements[3][1] = -(top + bottom) / (top - bottom);
    proj.elements[3][2] = -near_plane / (far_plane - near_plane);
    
    return view * proj;
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
            .cullMode = VK_CULL_MODE_NONE, 
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
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
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
                    .descriptorCount = max_descriptor_sets * 2,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .descriptorCount = max_descriptor_sets * 2,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = max_descriptor_sets * 3,
                },
            };
            
            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = max_descriptor_sets,
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
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 3,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 4,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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

    {
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };

        if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan texture sampler\n";
            veekay::app.running = false;
            return;
        }

        uint32_t pixels[] = {
            0xff000000, 0xffff00ff,
            0xffff00ff, 0xff000000,
        };

        missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                        VK_FORMAT_B8G8R8A8_UNORM,
                                                        pixels);
        
        floor_texture = loadTexture(cmd, "./assets/textures/ground.png");
        if (!floor_texture) floor_texture = missing_texture;
        floor_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);

        cone_texture = loadTexture(cmd, "./assets/textures/pyramid.png");
        if (!cone_texture) cone_texture = missing_texture;
        cone_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);

        cube_texture = loadTexture(cmd, "./assets/textures/jersi.png");
        if (!cube_texture) cube_texture = missing_texture;
        cube_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }

    // --- MESH GENERATION ---
    {
        // PLANE MESH (Floor)
        std::vector<Vertex> vertices = {
            {{-15.0f, 0.0f, -15.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{15.0f, 0.0f, -15.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{15.0f, 0.0f, 15.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-15.0f, 0.0f, 15.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            
            {{-15.0f, 0.0f, -15.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{15.0f, 0.0f, -15.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{15.0f, 0.0f, 15.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-15.0f, 0.0f, 15.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        };

        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0,
            4, 6, 5, 6, 4, 7
        };

        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        plane_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        plane_mesh.indices = uint32_t(indices.size());
    }

    {
        // CONE MESH
        cone_mesh = createConeMesh(64, 1.0f, 2.0f);
        // CUBE MESH
        cube_mesh = createCubeMesh(1.5f);
    }

    // --- MODEL SETUP ---
    models.clear();

    // 1. Пол (Индекс 0)
    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{
            .position = {0.0f, -2.0f, 0.0f} // Пол опущен
        },
        .material = Material{
            .albedo = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular = veekay::vec3{0.15f, 0.15f, 0.15f},
            .shininess = 8.0f,
            .texture = floor_texture,
            .sampler = floor_sampler,
        }
    });

    // 2. КУБ (Индекс 1)
    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {0.0f, 1.5f, 0.0f}, // Висит в центре
            .scale = {1.0f, 1.0f, 1.0f},
            .rotation = {0.0f, 0.0f, 0.0f}
        },
        .material = Material{
            .albedo = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular = veekay::vec3{1.0f, 1.0f, 1.0f},
            .shininess = 64.0f,
            .texture = cube_texture,
            .sampler = cube_sampler,
        }
    });

    // 3. Конусы (Индексы 2, 3, 4, 5)
    for (int i = 0; i < 4; ++i) {
        float axis_x = 0.5f + (i % 2) * 0.5f; 
        float axis_z = 0.2f + (i % 3) * 0.3f;
        models.emplace_back(Model{
            .mesh = cone_mesh,
            .transform = Transform{
                .position = {0.0f, 0.0f, 0.0f}, 
                .scale = {1.5f, 1.5f, 1.5f},
                .rotation = {0.0f, 0.0f, 0.0f}
            },
            .material = Material{
                .albedo = veekay::vec3{1.2f, 1.2f, 1.2f},
                .specular = veekay::vec3{1.0f, 1.0f, 1.0f},
                .shininess = 32.0f,
                .texture = cone_texture,
                .sampler = cone_sampler,
            },
            .rotation_axis = {axis_x, 1.0f, axis_z}
        });
    }

    // --- SHADOW SETUP ---
    {
        VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .extent = {
                .width = shadow_map_size,
                .height = shadow_map_size,
                .depth = 1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        
        if (vkCreateImage(device, &image_info, nullptr, &shadow_map_image) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image\n";
            veekay::app.running = false;
            return;
        }
        
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device, shadow_map_image, &mem_requirements);
        
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &mem_properties);
        
        uint32_t memory_type_index = UINT32_MAX;
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
            if ((mem_requirements.memoryTypeBits & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memory_type_index = i;
                break;
            }
        }
        
        VkMemoryAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };
        
        if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow_map_memory) != VK_SUCCESS) {
            std::cerr << "Failed to allocate shadow map memory\n";
            veekay::app.running = false;
            return;
        }
        
        vkBindImageMemory(device, shadow_map_image, shadow_map_memory, 0);
        
        VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = shadow_map_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        
        if (vkCreateImageView(device, &view_info, nullptr, &shadow_map_view) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image view\n";
            veekay::app.running = false;
            return;
        }
        
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .compareEnable = VK_TRUE,
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
            .unnormalizedCoordinates = VK_FALSE,
        };
        
        if (vkCreateSampler(device, &sampler_info, nullptr, &shadow_map_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map sampler\n";
            veekay::app.running = false;
            return;
        }
        
        light_space_buffer = new veekay::graphics::Buffer(
            sizeof(veekay::mat4),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        
        VkAttachmentDescription depth_attachment{
            .format = VK_FORMAT_D32_SFLOAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

        VkAttachmentReference depth_ref{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthStencilAttachment = &depth_ref,
        };

        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        VkRenderPassCreateInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &depth_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        if (vkCreateRenderPass(device, &render_pass_info, nullptr, &shadow_render_pass) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow render pass\n";
            veekay::app.running = false;
            return;
        }

        VkFramebufferCreateInfo framebuffer_info{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = shadow_render_pass,
            .attachmentCount = 1,
            .pAttachments = &shadow_map_view,
            .width = shadow_map_size,
            .height = shadow_map_size,
            .layers = 1,
        };

        if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &shadow_framebuffer) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow framebuffer\n";
            veekay::app.running = false;
            return;
        }
        
        std::cout << "Shadow mapping initialized successfully\n";
    }

    {
        shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
        if (!shadow_vertex_shader_module) {
            std::cerr << "Failed to load shadow vertex shader\n";
            veekay::app.running = false;
            return;
        }

        shadow_fragment_shader_module = loadShaderModule("./shaders/shadow.frag.spv");
        if (!shadow_fragment_shader_module) {
            std::cerr << "Failed to load shadow fragment shader\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo shadow_stages[2];
        shadow_stages[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shadow_vertex_shader_module,
            .pName = "main",
        };

        shadow_stages[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shadow_fragment_shader_module,
            .pName = "main",
        };

        VkVertexInputBindingDescription shadow_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        VkVertexInputAttributeDescription shadow_attribute{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        };

        VkPipelineVertexInputStateCreateInfo shadow_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &shadow_binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &shadow_attribute,
        };

        VkPipelineInputAssemblyStateCreateInfo shadow_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        VkViewport shadow_viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(shadow_map_size),
            .height = static_cast<float>(shadow_map_size),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D shadow_scissor{
            .offset = {0, 0},
            .extent = {shadow_map_size, shadow_map_size},
        };

        VkPipelineViewportStateCreateInfo shadow_viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &shadow_viewport,
            .scissorCount = 1,
            .pScissors = &shadow_scissor,
        };

        VkPipelineRasterizationStateCreateInfo shadow_raster{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_TRUE,
            .depthBiasConstantFactor = 1.25f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 1.75f,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo shadow_multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        VkPipelineDepthStencilStateCreateInfo shadow_depth{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };

        VkDescriptorSetLayoutBinding shadow_bindings[] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            },
        };

        VkDescriptorSetLayoutCreateInfo shadow_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = shadow_bindings,
        };

        if (vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr,
                                        &shadow_descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow descriptor set layout\n";
            veekay::app.running = false;
            return;
        }

        VkDescriptorSetAllocateInfo shadow_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &shadow_descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(device, &shadow_alloc, &shadow_descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate shadow descriptor set\n";
            veekay::app.running = false;
            return;
        }

        VkDescriptorBufferInfo shadow_buffer_infos[] = {
            {
                .buffer = light_space_buffer->buffer,
                .offset = 0,
                .range = sizeof(veekay::mat4),
            },
            {
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            },
        };

        VkWriteDescriptorSet shadow_writes[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow_descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &shadow_buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow_descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &shadow_buffer_infos[1],
            },
        };

        vkUpdateDescriptorSets(device, 2, shadow_writes, 0, nullptr);

        VkPipelineLayoutCreateInfo shadow_pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &shadow_descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(device, &shadow_pipeline_layout_info,
                                   nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow pipeline layout\n";
            veekay::app.running = false;
            return;
        }

        VkGraphicsPipelineCreateInfo shadow_pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shadow_stages,
            .pVertexInputState = &shadow_input,
            .pInputAssemblyState = &shadow_assembly,
            .pViewportState = &shadow_viewport_state,
            .pRasterizationState = &shadow_raster,
            .pMultisampleState = &shadow_multisample,
            .pDepthStencilState = &shadow_depth,
            .pColorBlendState = nullptr,
            .layout = shadow_pipeline_layout,
            .renderPass = shadow_render_pass,
            .subpass = 0,
        };

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                      &shadow_pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow pipeline\n";
            veekay::app.running = false;
            return;
        }

        std::cout << "Shadow pipeline created successfully\n";
    }

    for (Model& model : models) {
        Material& mat = model.material;
        
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(device, &alloc_info, &mat.descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set for material\n";
            veekay::app.running = false;
            return;
        }

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
        };

        VkDescriptorImageInfo image_info{
            .sampler = mat.sampler,
            .imageView = mat.texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo shadow_image_info{
            .sampler = shadow_map_sampler,
            .imageView = shadow_map_view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };
        
        VkDescriptorImageInfo shadow_image_info_raw{
            .sampler = missing_texture_sampler,
            .imageView = shadow_map_view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_infos[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &buffer_infos[1],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadow_image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadow_image_info_raw,
            },
        };

        vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                               write_infos, 0, nullptr);
    }
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    if (shadow_pipeline) {
        vkDestroyPipeline(device, shadow_pipeline, nullptr);
    }
    if (shadow_pipeline_layout) {
        vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
    }
    if (shadow_descriptor_set_layout) {
        vkDestroyDescriptorSetLayout(device, shadow_descriptor_set_layout, nullptr);
    }
    if (shadow_framebuffer) {
        vkDestroyFramebuffer(device, shadow_framebuffer, nullptr);
    }
    if (shadow_render_pass) {
        vkDestroyRenderPass(device, shadow_render_pass, nullptr);
    }
    if (shadow_fragment_shader_module) {
        vkDestroyShaderModule(device, shadow_fragment_shader_module, nullptr);
    }
    if (shadow_vertex_shader_module) {
        vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
    }

    if (light_space_buffer) {
        delete light_space_buffer;
    }
    if (shadow_map_sampler) {
        vkDestroySampler(device, shadow_map_sampler, nullptr);
    }
    if (shadow_map_view) {
        vkDestroyImageView(device, shadow_map_view, nullptr);
    }
    if (shadow_map_image) {
        vkDestroyImage(device, shadow_map_image, nullptr);
    }
    if (shadow_map_memory) {
        vkFreeMemory(device, shadow_map_memory, nullptr);
    }

    if (floor_texture && floor_texture != missing_texture) {
        delete floor_texture;
    }
    if (cone_texture && cone_texture != missing_texture) {
        delete cone_texture;
    }
    if (cube_texture && cube_texture != missing_texture) {
        delete cube_texture;
    }

    if (floor_sampler && floor_sampler != missing_texture_sampler) {
        vkDestroySampler(device, floor_sampler, nullptr);
    }
    if (cone_sampler && cone_sampler != missing_texture_sampler) {
        vkDestroySampler(device, cone_sampler, nullptr);
    }
    if (cube_sampler && cube_sampler != missing_texture_sampler) {
        vkDestroySampler(device, cube_sampler, nullptr);
    }
    
    if (missing_texture_sampler) {
        vkDestroySampler(device, missing_texture_sampler, nullptr);
    }
    if (missing_texture) {
        delete missing_texture;
    }

    delete cone_mesh.index_buffer;
    delete cone_mesh.vertex_buffer;

    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;

    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;

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
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene Configuration Panel", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "=== RENDER ENGINE v2.0 ===");
    ImGui::Text("Objects rendered: %zu", models.size());
    ImGui::Text("Trajectory: Chaos Mode");
    ImGui::Separator();
    
    ImGui::End();

    {
        using namespace veekay::input;

        if (mouse::isButtonDown(mouse::Button::left)) {
            auto move_delta = mouse::cursorDelta();
            camera.rotation.y -= move_delta.x * 0.1f;
            camera.rotation.x += move_delta.y * 0.1f;
            if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f;
            if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
        }

        auto view = camera.view();
        veekay::vec3 right = {view.elements[0][0], view.elements[1][0], view.elements[2][0]};
        veekay::vec3 up = {view.elements[0][1], view.elements[1][1], view.elements[2][1]};
        veekay::vec3 front = {view.elements[0][2], view.elements[1][2], view.elements[2][2]};

        if (keyboard::isKeyDown(keyboard::Key::w)) camera.position += front * 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::s)) camera.position -= front * 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::q)) camera.position += up * 0.1f;
        if (keyboard::isKeyDown(keyboard::Key::z)) camera.position -= up * 0.1f;
    }

    // --- ЛОГИКА ДЛЯ КУБА (models[1]) ---
    if (models.size() > 1) {
        Model& cube = models[1];
        // Куб медленно вращается в центре
        cube.transform.rotation.y += 20.0f * 0.016f;
        cube.transform.rotation.x += 10.0f * 0.016f;
        // Пульсация
        float t = float(time);
        float scale = 1.0f + sinf(t * 3.0f) * 0.1f;
        cube.transform.scale = {scale, scale, scale};
    }

    // --- ЛОГИКА ДЛЯ КОНУСОВ (Начинаем с 2, так как 0 - пол, 1 - куб) ---
    for (size_t i = 2; i < models.size(); ++i) {
        Model& cone = models[i];
        
        float offset = (float)i * 1.5f; 
        
        // Вращение
        cone.transform.rotation.x += (50.0f + offset * 5.0f) * 0.016f;
        cone.transform.rotation.y += (30.0f - offset * 2.0f) * 0.016f;
        
        float t = float(time);
        float speed = 0.8f + (i % 2) * 0.4f;
        float t_scaled = t * speed + offset;

        // Летаем вокруг куба (увеличили радиус до 4.0)
        float radius_x = 4.0f + (i % 2); 
        float radius_z = 4.0f + ((i + 1) % 2);

        cone.transform.position.x = sinf(t_scaled) * radius_x;
        cone.transform.position.z = sinf(t_scaled * 0.5f) * radius_z; 
        
        cone.transform.position.y = 3.0f + cosf(t_scaled * 2.0f) * 1.5f; 
    }
    
    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    veekay::vec3 light_pos = {6.0f, 10.0f, 6.0f};
    veekay::mat4 light_space_matrix = calculateLightSpaceMatrix(light_pos);

    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .view_position = camera.position,
        .directional_light = DirectionalLight{
            .direction = {0.2f, 1.0f, 0.5f}, 
            .ambient = {0.6f, 0.6f, 0.6f}, 
            .diffuse = {2.0f, 2.0f, 2.0f}, 
            .specular = {1.0f, 1.0f, 1.0f},
        },
        .light_space_matrix = light_space_matrix,
    };

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];
        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.material.albedo;
        uniforms.shininess = model.material.shininess;
        uniforms.specular_color = model.material.specular;
    }

    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
    *(veekay::mat4*)light_space_buffer->mapped_region = light_space_matrix;

    const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        const ModelUniforms& uniforms = model_uniforms[i];
        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    static bool first_frame = true;
    
    vkResetCommandBuffer(cmd, 0);

    {
        VkCommandBufferBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &info);
    }

    VkDeviceSize zero_offset = 0;
    const size_t model_uniforms_alignment =
        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = first_frame ? static_cast<VkAccessFlags>(0) : static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT),
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadow_map_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
            first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        
        VkRenderPassBeginInfo shadow_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = shadow_render_pass,
            .framebuffer = shadow_framebuffer,
            .renderArea = {
                .offset = {0, 0},
                .extent = {shadow_map_size, shadow_map_size},
            },
            .clearValueCount = 1,
            .pClearValues = &clear_depth,
        };

        vkCmdBeginRenderPass(cmd, &shadow_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0; i < models.size(); ++i) {
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

            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
                                    0, 1, &shadow_descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        
        VkImageMemoryBarrier read_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadow_map_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &read_barrier);
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        std::vector<size_t> render_order(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            render_order[i] = i;
        }
        
        std::sort(render_order.begin(), render_order.end(), 
            [](size_t a, size_t b) {
                veekay::vec3 delta_a = models[a].transform.position - camera.position;
                float dist_sq_a = delta_a.x * delta_a.x + delta_a.y * delta_a.y + delta_a.z * delta_a.z;
                
                veekay::vec3 delta_b = models[b].transform.position - camera.position;
                float dist_sq_b = delta_b.x * delta_b.x + delta_b.y * delta_b.y + delta_b.z * delta_b.z;
                
                return dist_sq_a > dist_sq_b;
            });

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t idx = 0; idx < render_order.size(); ++idx) {
            size_t i = render_order[idx];
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

            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &model.material.descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        std::cerr << "Failed to end command buffer recording\n";
        return;
    }
    
    first_frame = false;
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
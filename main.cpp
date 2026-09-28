#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <span>
#include <optional>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <print>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <chrono>
#include <unordered_set>
#include <nlohmann/json.hpp>

#include "common.hpp"
#include "cpu_engine.hpp"

template<typename T>
static std::vector<T> loadBinaryData(const std::string_view filepath) {
    std::ifstream file{std::string{filepath}, std::ios::binary | std::ios::ate};

    if (!file.is_open()) {
        throw std::runtime_error{"Failed to open: " + std::string{filepath}};
    }

    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::size_t elementsToRead = static_cast<std::size_t>(fileSize) / sizeof(T);
    std::vector<T> buffer(elementsToRead);

    if (elementsToRead > 0) {
        file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
    }

    return buffer;
}

class EvacuationEngine {
public:
    std::string vulkanNodesPath = "vulkan_nodes.bin";
    std::string vulkanEdgesPath = "vulkan_edges.bin";
    std::string vulkanCarsPath = "vulkan_cars.bin";
    bool isHeadless = false;
    bool useGpu = true;
    std::uint32_t cpuThreads = 0;
    float maxSimulationTime = 0.f;
    std::chrono::time_point<std::chrono::high_resolution_clock> benchmarkStartTime;
    std::optional<CPUEngine> cpuEngine;
    std::string outputFilePath = "results.json";
    std::vector<std::int32_t> configClosedExits;

    void loadConfig(const std::string &configPath) {
        std::ifstream f{configPath};
        if (!f.is_open()) {
            std::println("Could not open config file: {}. Using default settings.", configPath);
            isPaused = !isHeadless;
            return;
        }
        try {
            nlohmann::json data = nlohmann::json::parse(f);
            if (data.contains("vulkan_nodes_path")) {
                vulkanNodesPath = data["vulkan_nodes_path"].get<std::string>();
            }
            if (data.contains("vulkan_edges_path")) {
                vulkanEdgesPath = data["vulkan_edges_path"].get<std::string>();
            }
            if (data.contains("vulkan_cars_path")) {
                vulkanCarsPath = data["vulkan_cars_path"].get<std::string>();
            }
            if (data.contains("headless")) {
                isHeadless = data["headless"].get<bool>();
            }
            if (data.contains("use_gpu")) {
                useGpu = data["use_gpu"].get<bool>();
            }
            if (data.contains("cpu_threads")) {
                cpuThreads = data["cpu_threads"].get<std::uint32_t>();
            }
            if (data.contains("max_simulation_time")) {
                maxSimulationTime = data["max_simulation_time"].get<float>();
            }
            if (data.contains("output_file")) {
                outputFilePath = data["output_file"].get<std::string>();
            }
            if (data.contains("participation")) {
                participation = data["participation"].get<double>();
            }
            if (data.contains("closed_exits")) {
                configClosedExits = data["closed_exits"].get<std::vector<std::int32_t> >();
            }
            isPaused = !isHeadless;
            std::println("Configuration loaded from {}", configPath);
        } catch (const std::exception &e) {
            std::println("Error parsing config file: {}. Details: {}", configPath, e.what());
            isPaused = !isHeadless;
        }
    }

    void run() {
        if (!useGpu) {
            cpuEngine.emplace(cpuThreads);
            std::println("Initialized CPU Engine with {} worker threads.", cpuEngine->getNumThreads());
        }

        if (!isHeadless) {
            initWindow();
        }
        initVulkan();
        loadMapDataAndCreateBuffers();
        createComputePipeline();

        if (!isHeadless) {
            createSwapchain();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();

            initImGui();
        }

        mainLoop();
    }

    ~EvacuationEngine() {
        if (*device) {
            device.waitIdle();
            if (imguiInitialized) {
                ImGui_ImplVulkan_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImGui::DestroyContext();
            }
            // Release the presentation objects while the window still exists. The NVIDIA Wayland
            // driver dereferences the window's wl_surface when destroying the swapchain and surface,
            // so letting the member destructors run after glfwTerminate() segfaults on exit.
            framebuffers.clear();
            swapchainImageViews.clear();
            swapchain = nullptr;
            surface = nullptr;
        }
        if (window != nullptr) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

private:
    GLFWwindow *window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    std::uint32_t queueFamilyIndex = 0;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool commandPool = nullptr;

    vk::raii::Buffer nodeBuffer = nullptr;
    vk::raii::DeviceMemory nodeMemory = nullptr;
    vk::raii::Buffer edgeBuffer = nullptr;
    vk::raii::DeviceMemory edgeMemory = nullptr;
    vk::raii::Buffer carBuffer = nullptr;
    vk::raii::DeviceMemory carMemory = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline clearEdgesPipeline = nullptr;
    vk::raii::Pipeline buildGridPipeline = nullptr;
    vk::raii::Pipeline physicsPipeline = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorPool imguiPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;
    vk::raii::QueryPool timestampQueryPool = nullptr;
    bool hasTimestampQueries = true;
    double gpuClearMs = 0.0;
    double gpuGridMs = 0.0;
    double gpuPhysMs = 0.0;
    double gpuBarrMs = 0.0;
    double gpuTotalMs = 0.0;
    bool printedGpuTimestamps = false;

    std::uint32_t totalCars = 0;
    std::uint32_t totalEdges = 0;
    const std::uint32_t WIDTH = 800;
    const std::uint32_t HEIGHT = 600;

    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::raii::ImageView> swapchainImageViews;
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<vk::raii::Framebuffer> framebuffers;
    vk::Format swapchainImageFormat{};
    vk::Extent2D swapchainExtent;

    vk::raii::PipelineLayout graphicsPipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline streetPipeline = nullptr;
    vk::raii::Pipeline exitNodePipeline = nullptr;
    GraphicsConstants mapBounds;

    bool isPaused = true;
    std::int32_t simSpeed = 1;
    bool framebufferResized = false;

    bool isDragging = false;
    double lastMouseX = 0.;
    double lastMouseY = 0.;

    bool isSelecting = false;
    double startMouseX = 0.;
    double startMouseY = 0.;
    double currentMouseX = 0.;
    double currentMouseY = 0.;

    bool isInspecting = false;
    double inspectMouseX = 0.;
    double inspectMouseY = 0.;

    std::vector<GPU_Car> cpuCars;
    std::vector<GPU_Edge> cpuEdges;
    std::vector<GPU_Node> cpuNodes;

    vk::raii::Buffer carReadbackBuffer = nullptr;
    vk::raii::DeviceMemory carReadbackMemory = nullptr;
    vk::raii::Buffer edgeReadbackBuffer = nullptr;
    vk::raii::DeviceMemory edgeReadbackMemory = nullptr;
    vk::raii::Buffer nodeReadbackBuffer = nullptr;
    vk::raii::DeviceMemory nodeReadbackMemory = nullptr;

    std::int32_t selectedCarId = 0;
    bool imguiInitialized = false;

    std::int32_t statGarage = 0;
    std::int32_t statRoad = 0;
    std::int32_t statEvacuated = 0;
    std::int32_t statStuck = 0;
    std::int32_t statDisabled = 0;
    float statAvgSpeed = 0.f;
    float simTime = 0.f;
    double participation = 1.;
    bool hasStarted = false;

    std::vector<std::int32_t> flowrateHistory;
    std::vector<std::int32_t> evacuatedHistory;
    std::vector<std::int32_t> garageHistory;
    std::vector<std::int32_t> roadHistory;
    std::vector<std::int32_t> stuckHistory;
    std::vector<float> avgSpeedHistory;
    float lastRecordTime = 0.f;
    std::int32_t lastEvacuatedCount = 0;

    struct IncomingEdge {
        std::int32_t sourceNode = 0;
        std::int32_t edgeIdx = 0;
        float travelTime = 0.f;
    };

    std::vector<std::vector<IncomingEdge> > reverseGraph;
    std::vector<std::vector<std::int32_t> > forwardGraph;
    std::vector<std::int32_t> allExitNodes;
    std::vector<bool> isExitOpen;
    std::vector<std::vector<std::int32_t> > exitFlowrateHistory;
    std::vector<std::int32_t> exitLastEvacuatedCount;

    void recalculateGPS() {
        std::println("Recalculating City-Wide GPS Routes...");
        const auto startTime = std::chrono::high_resolution_clock::now();

        std::vector edge_car_counts(totalEdges, 0);
        std::vector edge_has_stuck(totalEdges, false);
        if (!cpuCars.empty()) {
            for (const auto &car: cpuCars) {
                if (car.state == CarState::Driving || car.state == CarState::Queuing || car.state == CarState::Stuck) {
                    if (car.current_edge_idx >= 0 && car.current_edge_idx < static_cast<std::int32_t>(totalEdges)) {
                        ++edge_car_counts[car.current_edge_idx];
                        if (car.state == CarState::Stuck) {
                            edge_has_stuck[car.current_edge_idx] = true;
                        }
                    }
                }
            }
        }

        for (auto &edge: cpuEdges) {
            edge.next_edge_idx = -1;
        }

        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            const std::int32_t nodeIdx = allExitNodes[i];
            cpuNodes[nodeIdx].type = isExitOpen[i] ? NodeType::OpenExit : NodeType::ClosedExit;
        }

        std::vector min_travel_time(cpuNodes.size(), std::numeric_limits<float>::max());
        std::vector next_node_to_exit(cpuNodes.size(), -1);

        using NodeRecord = std::pair<float, std::int32_t>; // {travel_time, node_idx}
        std::priority_queue<NodeRecord, std::vector<NodeRecord>, std::greater<> > pq;

        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            if (isExitOpen[i]) {
                const std::int32_t exit_idx = allExitNodes[i];
                min_travel_time[exit_idx] = 0.f;
                pq.emplace(0.f, exit_idx);
            }
        }

        while (!pq.empty()) {
            const auto [current_time, current_node] = pq.top();
            pq.pop();

            if (current_time > min_travel_time[current_node]) {
                continue;
            }

            for (const auto &incoming: reverseGraph[current_node]) {
                float travelTime = incoming.travelTime;
                if (!cpuCars.empty()) {
                    if (edge_has_stuck[incoming.edgeIdx]) {
                        travelTime = 1e6f;
                    } else {
                        const float capacity = std::max(cpuEdges[incoming.edgeIdx].length / 5.f, 1.f);
                        const float congestion = static_cast<float>(edge_car_counts[incoming.edgeIdx]) / capacity;
                        // BPR (Bureau of Public Roads) congestion function
                        travelTime = incoming.travelTime * (1.f + 4.f * std::pow(congestion, 4.f));
                    }
                }
                const float new_time = current_time + travelTime;

                if (new_time < min_travel_time[incoming.sourceNode]) {
                    min_travel_time[incoming.sourceNode] = new_time;
                    next_node_to_exit[incoming.sourceNode] = current_node;
                    pq.emplace(new_time, incoming.sourceNode);
                }
            }
        }

        for (std::int32_t i = 0; i < totalEdges; ++i) {
            const std::int32_t end_node = cpuEdges[i].end_node_idx;
            const std::int32_t target_node = next_node_to_exit[end_node];

            if (target_node != -1) {
                for (const std::int32_t j: forwardGraph[end_node]) {
                    if (cpuEdges[j].end_node_idx == target_node) {
                        cpuEdges[i].next_edge_idx = j;
                        break;
                    }
                }
            }
        }

        const auto endTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = endTime - startTime;
        std::println("Dijkstra Complete in {:.2f} ms!", ms.count());
    }

    void triggerDynamicReroute() {
        device.waitIdle();
        const bool wasPaused = isPaused;
        isPaused = true;

        recalculateGPS();

        {
            const vk::DeviceSize bufferSize = sizeof(GPU_Edge) * totalEdges;
            const auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            auto *mappedData = static_cast<GPU_Edge *>(stagingMemory.mapMemory(0, bufferSize));
            std::ranges::copy(cpuEdges, mappedData);
            stagingMemory.unmapMemory();
            copyBuffer(stagingBuffer, edgeBuffer, bufferSize);
        }

        {
            const vk::DeviceSize bufferSize = sizeof(GPU_Node) * cpuNodes.size();
            const auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            auto *mappedData = static_cast<GPU_Node *>(stagingMemory.mapMemory(0, bufferSize));
            std::ranges::copy(cpuNodes, mappedData);
            stagingMemory.unmapMemory();
            copyBuffer(stagingBuffer, nodeBuffer, bufferSize);
        }

        isPaused = wasPaused;
    }

    void recreateSwapchain() {
        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        framebuffers.clear();
        swapchainImageViews.clear();
        swapchain = nullptr;

        createSwapchain();
        createFramebuffers();

        mapBounds.aspect_ratio = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    }

    static void framebufferResizeCallback(GLFWwindow *window, const std::int32_t width, const std::int32_t height) {
        auto &engine = *static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(window));
        engine.framebufferResized = true;
    }

    void initImGui() {
        const std::array<vk::DescriptorPoolSize, 11> poolSizes = {
            {
                {vk::DescriptorType::eSampler, 1000},
                {vk::DescriptorType::eCombinedImageSampler, 1000},
                {vk::DescriptorType::eSampledImage, 1000},
                {vk::DescriptorType::eStorageImage, 1000},
                {vk::DescriptorType::eUniformTexelBuffer, 1000},
                {vk::DescriptorType::eStorageTexelBuffer, 1000},
                {vk::DescriptorType::eUniformBuffer, 1000},
                {vk::DescriptorType::eStorageBuffer, 1000},
                {vk::DescriptorType::eUniformBufferDynamic, 1000},
                {vk::DescriptorType::eStorageBufferDynamic, 1000},
                {vk::DescriptorType::eInputAttachment, 1000}
            }
        };

        const vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1000, .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        imguiPool = vk::raii::DescriptorPool{device, poolInfo};

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = *instance;
        init_info.PhysicalDevice = *physicalDevice;
        init_info.Device = *device;
        init_info.QueueFamily = queueFamilyIndex;
        init_info.Queue = *queue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = *imguiPool;
        init_info.MinImageCount = 2;
        init_info.ImageCount = static_cast<std::uint32_t>(swapchainImages.size());
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;
        init_info.PipelineInfoMain.RenderPass = *renderPass;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&init_info);
        imguiInitialized = true;
    }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        window = glfwCreateWindow(static_cast<std::int32_t>(WIDTH), static_cast<std::int32_t>(HEIGHT),
                                  "Vienna Evacuation Simulator", nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        glfwSetScrollCallback(window, [](GLFWwindow *win, const double xoffset, const double yoffset) {
            // ignore scroll if hovering over the UI
            if (ImGui::GetIO().WantCaptureMouse) {
                return;
            }

            auto &engine = *static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));
            if (yoffset > 0) {
                engine.mapBounds.zoom_level *= 1.15f;
            } else {
                engine.mapBounds.zoom_level /= 1.15f;
            }
            engine.mapBounds.zoom_level = std::clamp(engine.mapBounds.zoom_level, 0.5f, 500.f);
        });

        glfwSetMouseButtonCallback(
            window, [](GLFWwindow *win, const std::int32_t button, const std::int32_t action, const std::int32_t mods) {
                if (ImGui::GetIO().WantCaptureMouse) {
                    return;
                }

                auto &engine = *static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));

                if (button == GLFW_MOUSE_BUTTON_LEFT) {
                    if (action == GLFW_PRESS) {
                        if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                            glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                            engine.isSelecting = true;
                            glfwGetCursorPos(win, &engine.startMouseX, &engine.startMouseY);
                            engine.currentMouseX = engine.startMouseX;
                            engine.currentMouseY = engine.startMouseY;
                        } else {
                            engine.isDragging = true;
                            glfwGetCursorPos(win, &engine.lastMouseX, &engine.lastMouseY);
                        }
                    } else if (action == GLFW_RELEASE) {
                        if (engine.isSelecting) {
                            engine.isSelecting = false;
                            engine.applyMarqueeSelection();
                        }
                        engine.isDragging = false;
                    }
                }

                // inspect car
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    if (action == GLFW_PRESS) {
                        engine.isInspecting = true;
                        glfwGetCursorPos(win, &engine.inspectMouseX, &engine.inspectMouseY);

                        const auto [worldX, worldY] = engine.screenToWorld(engine.inspectMouseX, engine.inspectMouseY);
                        engine.takeSnapshot();
                        engine.selectClosestCar(worldX, worldY);
                    } else if (action == GLFW_RELEASE) {
                        engine.isInspecting = false;
                    }
                }
            });

        glfwSetCursorPosCallback(window, [](GLFWwindow *win, const double xpos, const double ypos) {
            auto &engine = *static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));

            if (engine.isSelecting) {
                engine.currentMouseX = xpos;
                engine.currentMouseY = ypos;
            } else if (engine.isInspecting) {
                engine.inspectMouseX = xpos;
                engine.inspectMouseY = ypos;
            } else if (engine.isDragging) {
                const double deltaX = xpos - engine.lastMouseX;
                const double deltaY = ypos - engine.lastMouseY;

                const float screenFactorX = engine.mapBounds.extent_height / static_cast<float>(engine.swapchainExtent.
                                                height);
                const float screenFactorY = engine.mapBounds.extent_height / static_cast<float>(engine.swapchainExtent.
                                                height);

                engine.mapBounds.camera_x -= static_cast<float>(deltaX) * (screenFactorX / engine.mapBounds.zoom_level);
                engine.mapBounds.camera_y += static_cast<float>(deltaY) * (screenFactorY / engine.mapBounds.zoom_level);

                engine.lastMouseX = xpos;
                engine.lastMouseY = ypos;
            }
        });

        glfwSetKeyCallback(
            window, [](GLFWwindow *win, const std::int32_t key, const std::int32_t scancode, const std::int32_t action,
                       const std::int32_t mods) {
                auto &engine = *static_cast<EvacuationEngine *>(glfwGetWindowUserPointer(win));

                if (action == GLFW_PRESS) {
                    if (key == GLFW_KEY_SPACE) {
                        engine.isPaused = !engine.isPaused;
                        if (engine.isPaused) {
                            engine.takeSnapshot();
                        }
                    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
                        engine.simSpeed = std::min(engine.simSpeed * 2, 256);
                    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
                        engine.simSpeed = std::max(engine.simSpeed / 2, 1);
                    }
                }
            });
    }

    void initVulkan() {
        context = vk::raii::Context{};

        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Vienna Evacuation", .applicationVersion = 1,
            .pEngineName = "No Engine", .engineVersion = 1, .apiVersion = VK_API_VERSION_1_3
        };

        std::uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = nullptr;
        if (!isHeadless) {
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        }

        const vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo, .enabledExtensionCount = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions
        };

        instance = vk::raii::Instance{context, createInfo};

        const vk::raii::PhysicalDevices physicalDevices{instance};
        physicalDevice = physicalDevices.front();

        const std::vector queueFamilies{physicalDevice.getQueueFamilyProperties()};
        for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics &&
                queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                queueFamilyIndex = i;
                break;
            }
        }

        constexpr float queuePriority = 1.f;
        const vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = queueFamilyIndex, .queueCount = 1, .pQueuePriorities = &queuePriority
        };

        std::vector<const char *> activeDeviceExtensions;
        if (!isHeadless) {
            activeDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
        activeDeviceExtensions.push_back("VK_KHR_shader_draw_parameters");

        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.vertexPipelineStoresAndAtomics = true;
        deviceFeatures.fragmentStoresAndAtomics = true;

        const vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1, .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(activeDeviceExtensions.size()),
            .ppEnabledExtensionNames = activeDeviceExtensions.data(), .pEnabledFeatures = &deviceFeatures
        };

        device = vk::raii::Device{physicalDevice, deviceCreateInfo};
        queue = vk::raii::Queue{device, queueFamilyIndex, 0};

        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndex
        };
        commandPool = vk::raii::CommandPool{device, poolInfo};

        const vk::QueryPoolCreateInfo queryPoolInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = 16
        };
        timestampQueryPool = vk::raii::QueryPool{device, queryPoolInfo};
    }

    [[nodiscard]] std::uint32_t findMemoryType(const std::uint32_t typeFilter,
                                               const vk::MemoryPropertyFlags properties) const {
        const vk::PhysicalDeviceMemoryProperties memProperties{physicalDevice.getMemoryProperties()};
        for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if (typeFilter & 1 << i && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error{"Failed to find suitable memory type!"};
    }

    [[nodiscard]] std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties) const {
        const vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
        vk::raii::Buffer newBuffer{device, bufferInfo};

        const vk::MemoryRequirements memRequirements{newBuffer.getMemoryRequirements()};
        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        vk::raii::DeviceMemory newMemory{device, allocInfo};

        newBuffer.bindMemory(*newMemory, 0);
        return {std::move(newBuffer), std::move(newMemory)};
    }

    void copyBuffer(const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer,
                    const vk::DeviceSize size) const {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffers commandBuffers{device, allocInfo};
        const vk::raii::CommandBuffer &cmdBuffer{commandBuffers.front()};

        constexpr vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        cmdBuffer.begin(beginInfo);
        const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
        cmdBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
        cmdBuffer.end();

        const vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmdBuffer};
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    template<typename T>
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> uploadVectorToGPU(std::span<const T> data) {
        const vk::DeviceSize bufferSize = sizeof(T) * data.size();
        const auto [stagingBuffer, stagingMemory] = createBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void *mappedData = stagingMemory.mapMemory(0, bufferSize);
        std::memcpy(mappedData, data.data(), static_cast<std::size_t>(bufferSize));
        stagingMemory.unmapMemory();

        auto [deviceLocalBuffer, deviceLocalMemory] = createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        copyBuffer(stagingBuffer, deviceLocalBuffer, bufferSize);
        return {std::move(deviceLocalBuffer), std::move(deviceLocalMemory)};
    }

    void applyParticipation() {
        device.waitIdle();

        auto rawCars = loadBinaryData<GPU_Car>(vulkanCarsPath);
        std::vector cars_spawned_per_edge(cpuEdges.size(), 0);
        std::vector target_per_edge(cpuEdges.size(), 0);

        double fractional_cars = 0.;
        for (std::size_t i = 0; i < cpuEdges.size(); ++i) {
            const double ideal_cars = static_cast<double>(cpuEdges[i].spawn_capacity) * participation;
            fractional_cars += ideal_cars;
            const auto spawn_now = static_cast<std::int32_t>(fractional_cars);
            target_per_edge[i] = spawn_now;
            fractional_cars -= static_cast<double>(spawn_now);
        }

        for (auto &c: rawCars) {
            c.next_car_idx = -1;

            const std::int32_t edge_idx = c.current_edge_idx;
            if (edge_idx >= 0 && edge_idx < cpuEdges.size()) {
                if (cars_spawned_per_edge[edge_idx] < target_per_edge[edge_idx]) {
                    ++cars_spawned_per_edge[edge_idx];
                } else {
                    c.state = CarState::Disabled;
                }
            }
        }

        std::tie(carBuffer, carMemory) = uploadVectorToGPU<GPU_Car>(rawCars);
        cpuCars = rawCars;
        updateDescriptorSets();
        takeSnapshot();
    }

    void loadMapDataAndCreateBuffers() {
        auto nodes = loadBinaryData<GPU_Node>(vulkanNodesPath);
        auto rawEdges = loadBinaryData<GPU_Edge>(vulkanEdgesPath);
        auto rawCars = loadBinaryData<GPU_Car>(vulkanCarsPath);

        for (auto &n: nodes) {
            n.lock = -1;
            // multiply X coordinates by cos(48.2082) to fix map projection distortion
            n.x *= 0.6664f;
        }

        for (auto &e: rawEdges) {
            e.head_car_idx = -1;
            e.garage_lock = -1;
        }
        for (auto &c: rawCars) {
            c.next_car_idx = -1;
        }

        totalCars = static_cast<std::uint32_t>(rawCars.size());
        totalEdges = static_cast<std::uint32_t>(rawEdges.size());

        std::tie(nodeBuffer, nodeMemory) = uploadVectorToGPU<GPU_Node>(nodes);
        std::tie(edgeBuffer, edgeMemory) = uploadVectorToGPU<GPU_Edge>(rawEdges);
        std::tie(carBuffer, carMemory) = uploadVectorToGPU<GPU_Car>(rawCars);

        float temp_min_x = std::numeric_limits<float>::max();
        float temp_max_x = std::numeric_limits<float>::lowest();
        float temp_min_y = std::numeric_limits<float>::max();
        float temp_max_y = std::numeric_limits<float>::lowest();

        for (const auto &n: nodes) {
            temp_min_x = std::min(temp_min_x, n.x);
            temp_max_x = std::max(temp_max_x, n.x);
            temp_min_y = std::min(temp_min_y, n.y);
            temp_max_y = std::max(temp_max_y, n.y);
        }

        const float width_meters = temp_max_x - temp_min_x;
        const float height_meters = temp_max_y - temp_min_y;

        mapBounds.camera_x = temp_min_x + width_meters * 0.5f;
        mapBounds.camera_y = temp_min_y + height_meters * 0.5f;
        mapBounds.zoom_level = 1.f;
        mapBounds.extent_width = width_meters;
        mapBounds.extent_height = height_meters;
        mapBounds.aspect_ratio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

        cpuCars = rawCars;
        cpuEdges = rawEdges;
        cpuNodes = nodes;

        auto createReadback = [&](const vk::DeviceSize size) {
            return createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent);
        };

        std::tie(carReadbackBuffer, carReadbackMemory) = createReadback(sizeof(GPU_Car) * totalCars);
        std::tie(edgeReadbackBuffer, edgeReadbackMemory) = createReadback(sizeof(GPU_Edge) * totalEdges);
        std::tie(nodeReadbackBuffer, nodeReadbackMemory) = createReadback(sizeof(GPU_Node) * nodes.size());

        reverseGraph.resize(cpuNodes.size());
        forwardGraph.resize(cpuNodes.size());
        std::unordered_set<std::int32_t> exitNodeSet;

        for (std::int32_t i = 0; i < totalEdges; ++i) {
            const GPU_Edge &edge = cpuEdges[i];
            const float travel_time = edge.length / std::max(edge.max_speed, 0.1f);

            reverseGraph[edge.end_node_idx].push_back({
                .sourceNode = edge.start_node_idx,
                .edgeIdx = i,
                .travelTime = travel_time
            });
            forwardGraph[edge.start_node_idx].push_back(i);
        }

        for (std::size_t i = 0; i < cpuNodes.size(); ++i) {
            if (cpuNodes[i].type == NodeType::OpenExit || cpuNodes[i].type == NodeType::ClosedExit) {
                exitNodeSet.insert(static_cast<std::int32_t>(i));
            }
        }

        allExitNodes.assign(exitNodeSet.begin(), exitNodeSet.end());
        isExitOpen.resize(allExitNodes.size(), true);
        exitFlowrateHistory.assign(allExitNodes.size(), std::vector<std::int32_t>());
        exitLastEvacuatedCount.assign(allExitNodes.size(), 0);

        for (const std::int32_t entry: configClosedExits) {
            if (entry >= 0 && entry < static_cast<std::int32_t>(allExitNodes.size())) {
                isExitOpen[entry] = false;
            }
            if (auto it = std::ranges::find(allExitNodes, entry); it != allExitNodes.end()) {
                const std::size_t idx = std::distance(allExitNodes.begin(), it);
                isExitOpen[idx] = false;
            }
        }

        recalculateGPS();

        std::tie(nodeBuffer, nodeMemory) = uploadVectorToGPU<GPU_Node>(cpuNodes);
        std::tie(edgeBuffer, edgeMemory) = uploadVectorToGPU<GPU_Edge>(cpuEdges);
    }

    void createComputePipeline() {
        const auto clearEdgesCode = loadBinaryData<std::uint32_t>("clear_edges.spv");
        const vk::ShaderModuleCreateInfo clearEdgesInfo{
            .codeSize = clearEdgesCode.size() * 4, .pCode = clearEdgesCode.data()
        };
        const vk::raii::ShaderModule clearEdgesShader{device, clearEdgesInfo};

        const auto buildGridCode = loadBinaryData<std::uint32_t>("build_grid.spv");
        const vk::ShaderModuleCreateInfo buildGridInfo{
            .codeSize = buildGridCode.size() * 4, .pCode = buildGridCode.data()
        };
        const vk::raii::ShaderModule buildGridShader{device, buildGridInfo};

        const auto physicsCode = loadBinaryData<std::uint32_t>("physics.spv");
        const vk::ShaderModuleCreateInfo physShaderInfo{
            .codeSize = physicsCode.size() * 4, .pCode = physicsCode.data()
        };
        const vk::raii::ShaderModule physShader{device, physShaderInfo};

        constexpr std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            {
                {
                    .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                },
                {
                    .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                },
                {
                    .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eAll
                }
            }
        };

        const vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()), .pBindings = bindings.data()
        };
        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout{device, layoutInfo};

        constexpr vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(PushConstants)
        };

        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &*computeDescriptorSetLayout, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };
        computePipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

        const auto mainEntryPoint = "main";

        const vk::ComputePipelineCreateInfo cePipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *clearEdgesShader, .pName = mainEntryPoint},
            .layout = *computePipelineLayout
        };
        clearEdgesPipeline = vk::raii::Pipeline{device, nullptr, cePipelineInfo};

        const vk::ComputePipelineCreateInfo bgPipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *buildGridShader, .pName = mainEntryPoint},
            .layout = *computePipelineLayout
        };
        buildGridPipeline = vk::raii::Pipeline{device, nullptr, bgPipelineInfo};

        const vk::ComputePipelineCreateInfo physPipelineInfo{
            .flags = {},
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *physShader, .pName = mainEntryPoint},
            .layout = *computePipelineLayout
        };
        physicsPipeline = vk::raii::Pipeline{device, nullptr, physPipelineInfo};

        constexpr vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 3};
        const vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = 1, .poolSizeCount = 1,
            .pPoolSizes = &poolSize
        };
        descriptorPool = vk::raii::DescriptorPool{device, poolInfo};

        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &*computeDescriptorSetLayout
        };
        computeDescriptorSets = vk::raii::DescriptorSets{device, allocInfo};

        updateDescriptorSets();
    }

    void updateDescriptorSets() const {
        const vk::DescriptorBufferInfo nodeBufferInfo{.buffer = *nodeBuffer, .offset = 0, .range = vk::WholeSize};
        const vk::DescriptorBufferInfo edgeBufferInfo{.buffer = *edgeBuffer, .offset = 0, .range = vk::WholeSize};
        const vk::DescriptorBufferInfo carBufferInfo{.buffer = *carBuffer, .offset = 0, .range = vk::WholeSize};

        const std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {
            {
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 0, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &nodeBufferInfo
                },
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 1, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &edgeBufferInfo
                },
                {
                    .dstSet = *computeDescriptorSets[0], .dstBinding = 2, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &carBufferInfo
                }
            }
        };
        device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    void createSwapchain() {
        if (!*surface) {
            VkSurfaceKHR c_surface;
            if (glfwCreateWindowSurface(*instance, window, nullptr, &c_surface) != VK_SUCCESS)
                throw std::runtime_error{"Failed to create window surface!"};
            surface = vk::raii::SurfaceKHR{instance, c_surface};
        }

        const vk::SurfaceCapabilitiesKHR capabilities{physicalDevice.getSurfaceCapabilitiesKHR(*surface)};
        const std::vector formats{physicalDevice.getSurfaceFormatsKHR(*surface)};

        vk::SurfaceFormatKHR surfaceFormat = formats[0];
        for (const auto &availableFormat: formats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace ==
                vk::ColorSpaceKHR::eSrgbNonlinear) {
                surfaceFormat = availableFormat;
                break;
            }
        }
        swapchainImageFormat = surfaceFormat.format;

        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        swapchainExtent = vk::Extent2D{
            std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height)
        };

        std::uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        const vk::SwapchainCreateInfoKHR createInfo{
            .flags = {}, .surface = *surface, .minImageCount = imageCount, .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace, .imageExtent = swapchainExtent, .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform, .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = vk::PresentModeKHR::eFifo, .clipped = true
        };

        swapchain = vk::raii::SwapchainKHR{device, createInfo};
        swapchainImages = swapchain.getImages();

        for (const auto &image: swapchainImages) {
            constexpr vk::ComponentMapping components{
                .r = vk::ComponentSwizzle::eIdentity, .g = vk::ComponentSwizzle::eIdentity,
                .b = vk::ComponentSwizzle::eIdentity, .a = vk::ComponentSwizzle::eIdentity
            };
            constexpr vk::ImageSubresourceRange subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0,
                .layerCount = 1
            };
            const vk::ImageViewCreateInfo viewInfo{
                .flags = {}, .image = image, .viewType = vk::ImageViewType::e2D, .format = swapchainImageFormat,
                .components = components, .subresourceRange = subresourceRange
            };
            swapchainImageViews.emplace_back(device, viewInfo);
        }

        mapBounds.aspect_ratio = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    }

    void createRenderPass() {
        const vk::AttachmentDescription colorAttachment{
            .flags = {}, .format = swapchainImageFormat, .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined, .finalLayout = vk::ImageLayout::ePresentSrcKHR
        };
        constexpr vk::AttachmentReference colorAttachmentRef{
            .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal
        };
        const vk::SubpassDescription subpass{
            .flags = {}, .pipelineBindPoint = vk::PipelineBindPoint::eGraphics, .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef
        };
        const vk::RenderPassCreateInfo renderPassInfo{
            .flags = {}, .attachmentCount = 1, .pAttachments = &colorAttachment, .subpassCount = 1,
            .pSubpasses = &subpass
        };

        renderPass = vk::raii::RenderPass{device, renderPassInfo};
    }

    void createFramebuffers() {
        for (const auto &imageView: swapchainImageViews) {
            const std::array attachments = {*imageView};
            const vk::FramebufferCreateInfo framebufferInfo{
                .flags = {}, .renderPass = *renderPass, .attachmentCount = 1, .pAttachments = attachments.data(),
                .width = swapchainExtent.width, .height = swapchainExtent.height, .layers = 1
            };
            framebuffers.emplace_back(device, framebufferInfo);
        }
    }

    void createGraphicsPipeline() {
        const auto vertCode = loadBinaryData<std::uint32_t>("graphics_vert.spv");
        const auto fragCode = loadBinaryData<std::uint32_t>("graphics_frag.spv");

        const vk::raii::ShaderModule vertModule{
            device, vk::ShaderModuleCreateInfo{.codeSize = vertCode.size() * 4, .pCode = vertCode.data()}
        };
        const vk::raii::ShaderModule fragModule{
            device, vk::ShaderModuleCreateInfo{.codeSize = fragCode.size() * 4, .pCode = fragCode.data()}
        };

        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragModule, .pName = "main"
            }
        };

        constexpr vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        constexpr std::array dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor
        };
        const vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .dynamicStateCount = 2, .pDynamicStates = dynamicStates.data()
        };

        constexpr vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
            .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.f
        };
        constexpr vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1
        };
        constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1, .pAttachments = &colorBlendAttachment
        };
        constexpr vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = false, .depthWriteEnable = false, .depthCompareOp = vk::CompareOp::eLessOrEqual,
            .stencilTestEnable = false
        };

        constexpr vk::PushConstantRange pushRange{
            .stageFlags = vk::ShaderStageFlagBits::eAllGraphics, .offset = 0, .size = sizeof(GraphicsConstants)
        };
        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &*computeDescriptorSetLayout, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange
        };
        graphicsPipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

        const vk::GraphicsPipelineCreateInfo pipelineInfo{
            .flags = {}, .stageCount = 2, .pStages = shaderStages.data(), .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState, .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling, .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicStateInfo, .layout = *graphicsPipelineLayout, .renderPass = *renderPass,
            .subpass = 0
        };
        graphicsPipeline = vk::raii::Pipeline{device, nullptr, pipelineInfo};

        const auto streetVertCode = loadBinaryData<std::uint32_t>("streets_vert.spv");
        const auto streetFragCode = loadBinaryData<std::uint32_t>("streets_frag.spv");
        const vk::raii::ShaderModule streetVertModule{
            device, vk::ShaderModuleCreateInfo{.codeSize = streetVertCode.size() * 4, .pCode = streetVertCode.data()}
        };
        const vk::raii::ShaderModule streetFragModule{
            device, vk::ShaderModuleCreateInfo{.codeSize = streetFragCode.size() * 4, .pCode = streetFragCode.data()}
        };

        const std::array streetShaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *streetVertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *streetFragModule, .pName = "main"
            }
        };

        constexpr vk::PipelineInputAssemblyStateCreateInfo lineAssembly{.topology = vk::PrimitiveTopology::eLineList};

        vk::GraphicsPipelineCreateInfo streetPipelineInfo = pipelineInfo;
        streetPipelineInfo.pStages = streetShaderStages.data();
        streetPipelineInfo.pInputAssemblyState = &lineAssembly;

        streetPipeline = vk::raii::Pipeline{device, nullptr, streetPipelineInfo};

        const auto exitNodeVertCode = loadBinaryData<std::uint32_t>("exit_nodes_vert.spv");
        const auto exitNodeFragCode = loadBinaryData<std::uint32_t>("exit_nodes_frag.spv");
        const vk::raii::ShaderModule exitNodeVertModule{
            device,
            vk::ShaderModuleCreateInfo{.codeSize = exitNodeVertCode.size() * 4, .pCode = exitNodeVertCode.data()}
        };
        const vk::raii::ShaderModule exitNodeFragModule{
            device,
            vk::ShaderModuleCreateInfo{.codeSize = exitNodeFragCode.size() * 4, .pCode = exitNodeFragCode.data()}
        };

        const std::array exitNodeShaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eVertex, .module = *exitNodeVertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .flags = {}, .stage = vk::ShaderStageFlagBits::eFragment, .module = *exitNodeFragModule, .pName = "main"
            }
        };

        vk::GraphicsPipelineCreateInfo exitNodePipelineInfo = pipelineInfo;
        exitNodePipelineInfo.pStages = exitNodeShaderStages.data();

        exitNodePipeline = vk::raii::Pipeline{device, nullptr, exitNodePipelineInfo};
    }

    void mainLoop() {
        const vk::CommandBufferAllocateInfo cmdAllocInfo{
            .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffers cmdBuffers{device, cmdAllocInfo};
        const vk::raii::CommandBuffer &cmd{cmdBuffers.front()};

        const vk::raii::Semaphore imageAvailableSemaphore{device, vk::SemaphoreCreateInfo{}};
        const vk::raii::Semaphore renderFinishedSemaphore{device, vk::SemaphoreCreateInfo{}};
        constexpr vk::ClearValue clearColor{
            .color = vk::ClearColorValue{std::array{0.05f, 0.05f, 0.1f, 1.f}}
        };

        auto lastFrameTime = std::chrono::high_resolution_clock::now();
        double accumulatedSimTimeQueue = 0.;
        float lastSnapshotSimTime = simTime;

        takeSnapshot();

        if (!isPaused && !hasStarted) {
            applyParticipation();
            hasStarted = true;
        }

        benchmarkStartTime = std::chrono::high_resolution_clock::now();

        while (true) {
            if (!isHeadless) {
                glfwPollEvents();
                if (glfwWindowShouldClose(window)) {
                    break;
                }
            } else {
                if (simTime > 0.f && statRoad == 0) {
                    std::println("Simulation finished: no cars on the road or all cars on the road stuck.");
                    break;
                }
                if (maxSimulationTime > 0.f && simTime >= maxSimulationTime) {
                    std::println("Simulation finished: reached max_simulation_time of {:.1f}s.", maxSimulationTime);
                    break;
                }
            }

            const auto currentFrameTime = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> dt_duration = currentFrameTime - lastFrameTime;
            const double dt_real = dt_duration.count();
            lastFrameTime = currentFrameTime;

            if (!isPaused && simTime - lastSnapshotSimTime >= 60.f) {
                takeSnapshot();
                recalculateGPS();

                if (useGpu) {
                    const vk::DeviceSize bufferSize = sizeof(GPU_Edge) * totalEdges;
                    const auto [stagingBuffer, stagingMemory] = createBuffer(
                        bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                    );
                    void *mappedData = stagingMemory.mapMemory(0, bufferSize);
                    std::memcpy(mappedData, cpuEdges.data(), bufferSize);
                    stagingMemory.unmapMemory();
                    copyBuffer(stagingBuffer, edgeBuffer, bufferSize);
                }

                recordPeriodicHistory();

                lastSnapshotSimTime = simTime;

                if (isHeadless) {
                    std::println("Time: {:.1f}s | Garage: {} | Road: {} | Evacuated: {} | Stuck: {}",
                                 simTime, statGarage, statRoad, statEvacuated, statStuck);
                } else {
                    if (statRoad == 0 && statGarage == 0) {
                        isPaused = true;
                    }
                }
            }

            if (!isHeadless && !isPaused && selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars)
                && !cpuCars.empty()) {
                {
                    const vk::DeviceSize offset = sizeof(GPU_Car) * selectedCarId;
                    const void *mappedData = carReadbackMemory.mapMemory(offset, sizeof(GPU_Car));
                    std::memcpy(&cpuCars[selectedCarId], mappedData, sizeof(GPU_Car));
                    carReadbackMemory.unmapMemory();
                }

                const std::int32_t edgeIdx = cpuCars[selectedCarId].current_edge_idx;
                if (edgeIdx >= 0 && edgeIdx < static_cast<std::int32_t>(totalEdges)) {
                    const vk::DeviceSize offset = sizeof(GPU_Edge) * edgeIdx;
                    const void *mappedData = edgeReadbackMemory.mapMemory(offset, sizeof(GPU_Edge));
                    std::memcpy(&cpuEdges[edgeIdx], mappedData, sizeof(GPU_Edge));
                    edgeReadbackMemory.unmapMemory();

                    const std::int32_t nodeIdx = cpuEdges[edgeIdx].end_node_idx;
                    if (nodeIdx >= 0 && nodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                        const vk::DeviceSize nodeOffset = sizeof(GPU_Node) * nodeIdx;
                        const void *mappedNodeData = nodeReadbackMemory.mapMemory(nodeOffset, sizeof(GPU_Node));
                        std::memcpy(&cpuNodes[nodeIdx], mappedNodeData, sizeof(GPU_Node));
                        nodeReadbackMemory.unmapMemory();
                    }
                }
            }

            if (!isHeadless) {
                if (framebufferResized) {
                    framebufferResized = false;
                    recreateSwapchain();
                }
            }

            std::uint32_t imageIndex = 0;
            if (!isHeadless) {
                try {
                    auto [result, index] = swapchain.acquireNextImage(std::numeric_limits<std::uint64_t>::max(),
                                                                      *imageAvailableSemaphore, nullptr);
                    if (result == vk::Result::eSuboptimalKHR) {
                        framebufferResized = true;
                    }
                    imageIndex = index;
                } catch (const vk::OutOfDateKHRError &) {
                    framebufferResized = false;
                    recreateSwapchain();
                    continue;
                }

                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
            }

            if (!isHeadless) {
                ImGui::Begin("Simulation Control Panel");
                ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                ImGui::Separator();

                ImGui::Separator();
                ImGui::Text("--- EMERGENCY SCENARIO CONTROL ---");

                bool routingChanged = false;

                if (ImGui::Button("Open All Exits")) {
                    std::ranges::fill(isExitOpen, true);
                    routingChanged = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Close All Exits")) {
                    std::ranges::fill(isExitOpen, false);
                    routingChanged = true;
                }

                ImGui::Spacing();

                ImGui::BeginDisabled(hasStarted);
                auto participationFloat = static_cast<float>(participation);
                if (ImGui::SliderFloat("Participation", &participationFloat, 0.f, 1.f, "%.2f")) {
                    participation = static_cast<double>(participationFloat);
                }
                ImGui::EndDisabled();

                ImGui::Spacing();

                if (routingChanged) {
                    triggerDynamicReroute();
                }

                ImGui::Separator();

                if (ImGui::Button(isPaused ? "Play Simulation" : "Pause Simulation", ImVec2(-1.f, 40.f))) {
                    if (!hasStarted) {
                        applyParticipation();
                        hasStarted = true;
                    }
                    isPaused = !isPaused;
                    if (isPaused) {
                        takeSnapshot();
                    }
                }
                ImGui::Spacing();
                ImGui::SliderInt("Simulation Speed", &simSpeed, 1, 256, "%d x");

                ImGui::Separator();
                ImGui::Text("--- EVACUATION METRICS ---");
                {
                    std::int32_t hours = static_cast<std::int32_t>(simTime) / 3600;
                    std::int32_t minutes = static_cast<std::int32_t>(simTime) % 3600 / 60;
                    float seconds = std::fmod(simTime, 60.f);
                    ImGui::Text("Simulation Time: %02d:%02d:%04.1f", hours, minutes, seconds);
                }
                ImGui::Text("Waiting in Garage: %d", statGarage);
                ImGui::Text("Active on Road: %d", statRoad);
                ImGui::Text("Safely Evacuated: %d", statEvacuated);
                ImGui::Text("Stuck Cars: %d", statStuck);
                ImGui::Text("City Average Speed: %.1f km/h", statAvgSpeed * 3.6f);
                if (useGpu && gpuTotalMs > 0.0) {
                    ImGui::Spacing();
                    ImGui::Text("--- GPU HARDWARE PROFILING ---");
                    ImGui::Text("Step Compute: %.3f ms (%.1f RTF)", gpuTotalMs, 100.0 / gpuTotalMs);
                    ImGui::Text("  Clear Edges:  %.3f ms (%.1f%%)", gpuClearMs, (gpuClearMs / gpuTotalMs) * 100.0);
                    ImGui::Text("  Build Grid:   %.3f ms (%.1f%%)", gpuGridMs, (gpuGridMs / gpuTotalMs) * 100.0);
                    ImGui::Text("  Physics IDM:  %.3f ms (%.1f%%)", gpuPhysMs, (gpuPhysMs / gpuTotalMs) * 100.0);
                    ImGui::Text("  Barriers/Sync:%.3f ms (%.1f%%)", gpuBarrMs, (gpuBarrMs / gpuTotalMs) * 100.0);
                }

                std::int32_t participatingCars = static_cast<std::int32_t>(totalCars) - statDisabled;
                float progress = participatingCars > 0
                                     ? static_cast<float>(statEvacuated) / static_cast<float>(participatingCars)
                                     : 0.f;
                std::string progressText = std::format("Evacuation Progress: {:.1f}%", progress * 100.f);
                ImGui::ProgressBar(progress, ImVec2(-1.f, 0.f), progressText.c_str());

                if (!flowrateHistory.empty()) {
                    ImGui::Spacing();
                    ImGui::Text("--- CHARTS ---");
                    std::string flowText = std::format("{} cars/min", flowrateHistory.back());
                    std::vector<float> flowrateHistoryFloat;
                    flowrateHistoryFloat.reserve(flowrateHistory.size());
                    for (const auto val: flowrateHistory) {
                        flowrateHistoryFloat.push_back(static_cast<float>(val));
                    }
                    ImGui::PlotHistogram("Flowrate (cars/min)", flowrateHistoryFloat.data(),
                                         static_cast<std::int32_t>(flowrateHistoryFloat.size()),
                                         0, flowText.c_str(), 0.f, std::numeric_limits<float>::max(),
                                         ImVec2(0.f, 120.f));

                    std::string evacText = std::format("{} evacuated", evacuatedHistory.back());
                    std::vector<float> evacuatedHistoryFloat;
                    evacuatedHistoryFloat.reserve(evacuatedHistory.size());
                    for (const auto val: evacuatedHistory) {
                        evacuatedHistoryFloat.push_back(static_cast<float>(val));
                    }
                    ImGui::PlotLines("Total Evacuated", evacuatedHistoryFloat.data(),
                                     static_cast<std::int32_t>(evacuatedHistoryFloat.size()),
                                     0, evacText.c_str(), 0.f, std::numeric_limits<float>::max(), ImVec2(0.f, 120.f));
                }

                if (!cpuCars.empty()) {
                    ImGui::Spacing();
                    selectedCarId = std::clamp(selectedCarId, 0, static_cast<std::int32_t>(totalCars) - 1);

                    GPU_Car &car = cpuCars[selectedCarId];

                    ImGui::BeginChild("CarData", ImVec2(0, 150), true);
                    ImGui::Text("--- CAR %d ---", selectedCarId);

                    std::string carState;

                    switch (car.state) {
                        case CarState::Driving:
                            carState = "Driving";
                            break;
                        case CarState::Queuing:
                            carState = "Queuing at Intersection";
                            break;
                        case CarState::Evacuated:
                            carState = "Evacuated!";
                            break;
                        case CarState::Garage:
                            carState = "In Garage";
                            break;
                        case CarState::Stuck:
                            carState = "Stuck";
                            break;
                        case CarState::Disabled:
                            carState = "Disabled";
                            break;
                        default:
                            carState = "Unknown";
                    }

                    ImGui::Text("State: %s (%d)", carState.c_str(), car.state);
                    ImGui::Text("Speed: %.2f m/s (%.1f km/h)", car.speed, car.speed * 3.6f);
                    ImGui::Text("Position: %.2f meters", car.position);
                    ImGui::Text("Current Edge ID: %d", car.current_edge_idx);
                    ImGui::Text("Next Car in Linked List: %d", car.next_car_idx);
                    ImGui::EndChild();

                    if (car.current_edge_idx != -1) {
                        GPU_Edge &edge = cpuEdges[car.current_edge_idx];
                        ImGui::BeginChild("EdgeData", ImVec2(0, 140), true);
                        ImGui::Text("--- EDGE %d ---", car.current_edge_idx);
                        ImGui::Text("Length: %.2f meters", edge.length);
                        ImGui::Text("Max Speed: %.1f km/h", edge.max_speed * 3.6f);
                        ImGui::Text("Head Car ID: %d", edge.head_car_idx);
                        ImGui::Text("Target Node ID: %d", edge.end_node_idx);
                        ImGui::Text("Spawn Capacity: %d", edge.spawn_capacity);
                        ImGui::EndChild();
                    }
                }
                ImGui::End();

                if (isSelecting) {
                    ImDrawList &drawList = *ImGui::GetForegroundDrawList();
                    const ImVec2 p_min{
                        static_cast<float>(std::min(startMouseX, currentMouseX)),
                        static_cast<float>(std::min(startMouseY, currentMouseY))
                    };
                    const ImVec2 p_max{
                        static_cast<float>(std::max(startMouseX, currentMouseX)),
                        static_cast<float>(std::max(startMouseY, currentMouseY))
                    };
                    drawList.AddRectFilled(p_min, p_max, IM_COL32(0, 150, 255, 60), 0.f);
                    drawList.AddRect(p_min, p_max, IM_COL32(0, 150, 255, 255), 0.f, 0, 2.f);
                }

                if (isInspecting) {
                    ImDrawList &drawList = *ImGui::GetForegroundDrawList();
                    const ImVec2 center{static_cast<float>(inspectMouseX), static_cast<float>(inspectMouseY)};
                    const float r = worldDistanceToPixels(20.f / 111300.f);
                    drawList.AddCircleFilled(center, r, IM_COL32(255, 165, 0, 40), 64);
                    drawList.AddCircle(center, r, IM_COL32(255, 165, 0, 180), 64, 2.f);
                }

                ImGui::Render();
            }

            constexpr vk::CommandBufferBeginInfo cmdBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
            cmd.begin(cmdBeginInfo);

            std::int32_t stepsToRun = 0;
            if (isHeadless) {
                float timeToNextSnapshot = 60.f - (simTime - lastSnapshotSimTime);
                if (maxSimulationTime > 0.f) {
                    const float timeToMax = maxSimulationTime - simTime;
                    timeToNextSnapshot = std::min(timeToNextSnapshot, timeToMax);
                }
                stepsToRun = std::clamp(static_cast<std::int32_t>(std::lround(timeToNextSnapshot * 10.f)), 0, 600);
                if (stepsToRun == 0 && maxSimulationTime > 0.f && simTime >= maxSimulationTime) {
                    break;
                }
            } else {
                if (!isPaused) {
                    const double capped_dt = std::min(dt_real, 0.033);
                    accumulatedSimTimeQueue += capped_dt * simSpeed;
                    while (accumulatedSimTimeQueue >= 0.1) {
                        ++stepsToRun;
                        accumulatedSimTimeQueue -= 0.1;
                    }
                    if (stepsToRun > 256) {
                        stepsToRun = 256;
                        accumulatedSimTimeQueue = 0.;
                    }
                }
            }

            if (stepsToRun > 0) {
                simTime += static_cast<float>(stepsToRun) * 0.1f;

                if (!useGpu) {
                    for (std::int32_t step = 0; step < stepsToRun; ++step) {
                        cpuEngine->step(cpuNodes, cpuEdges, cpuCars, 0.1f);
                    }

                    if (!isHeadless) {
                        const vk::DeviceSize bufferSize = sizeof(GPU_Car) * totalCars;
                        const auto [stagingBuffer, stagingMemory] = createBuffer(
                            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                        );
                        void *mappedData = stagingMemory.mapMemory(0, bufferSize);
                        std::memcpy(mappedData, cpuCars.data(), bufferSize);
                        stagingMemory.unmapMemory();
                        copyBuffer(stagingBuffer, carBuffer, bufferSize);
                    }
                } else {
                    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                           {*computeDescriptorSets[0]}, nullptr);
                    const PushConstants pushData{.dt = 0.1f, .num_cars = totalCars, .num_edges = totalEdges};
                    cmd.pushConstants<PushConstants>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                                     pushData);

                    for (std::int32_t step = 0; step < stepsToRun; ++step) {
                        if (step == 0 && hasTimestampQueries) {
                            cmd.resetQueryPool(*timestampQueryPool, 0, 8);
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampQueryPool, 0);
                        }

                        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *clearEdgesPipeline);
                        cmd.dispatch((totalEdges + 255) / 256, 1, 1);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 1);
                        }

                        const vk::BufferMemoryBarrier edgeBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                            .buffer = *edgeBuffer, .offset = 0, .size = vk::WholeSize
                        };
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, edgeBarrier,
                                            nullptr);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 2);
                        }

                        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *buildGridPipeline);
                        cmd.dispatch((totalCars + 255) / 256, 1, 1);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 3);
                        }

                        const vk::BufferMemoryBarrier carBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                            .buffer = *carBuffer, .offset = 0, .size = vk::WholeSize
                        };
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, carBarrier,
                                            nullptr);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 4);
                        }

                        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *physicsPipeline);
                        cmd.dispatch((totalCars + 255) / 256, 1, 1);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 5);
                        }

                        const vk::PipelineStageFlags dstStage = isHeadless || step < stepsToRun - 1
                                                                    ? vk::PipelineStageFlagBits::eComputeShader
                                                                    : vk::PipelineStageFlagBits::eVertexShader;
                        constexpr vk::MemoryBarrier stepBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
                        };
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, dstStage, {}, stepBarrier,
                                            nullptr,
                                            nullptr);

                        if (step == 0 && hasTimestampQueries) {
                            cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampQueryPool, 6);
                        }
                    }
                }

                if (!isHeadless) {
                    const std::int32_t inspectEdgeIdx = selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>
                                                        (totalCars) && !cpuCars.empty()
                                                            ? cpuCars[selectedCarId].current_edge_idx
                                                            : -1;
                    const std::int32_t inspectNodeIdx = inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<
                                                            std::int32_t>(totalEdges)
                                                            ? cpuEdges[inspectEdgeIdx].end_node_idx
                                                            : -1;

                    std::vector<vk::BufferMemoryBarrier> barriers;
                    if (selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars)) {
                        barriers.push_back(vk::BufferMemoryBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead, .buffer = *carBuffer,
                            .offset = sizeof(GPU_Car) * selectedCarId, .size = sizeof(GPU_Car)
                        });
                    }
                    if (inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<std::int32_t>(totalEdges)) {
                        barriers.push_back(vk::BufferMemoryBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead, .buffer = *edgeBuffer,
                            .offset = sizeof(GPU_Edge) * inspectEdgeIdx, .size = sizeof(GPU_Edge)
                        });
                    }
                    if (inspectNodeIdx >= 0 && inspectNodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                        barriers.push_back(vk::BufferMemoryBarrier{
                            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead, .buffer = *nodeBuffer,
                            .offset = sizeof(GPU_Node) * inspectNodeIdx, .size = sizeof(GPU_Node)
                        });
                    }

                    if (!barriers.empty()) {
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eTransfer, {}, nullptr, barriers, nullptr);
                    }

                    if (selectedCarId >= 0 && selectedCarId < static_cast<std::int32_t>(totalCars)) {
                        cmd.copyBuffer(*carBuffer, *carReadbackBuffer, vk::BufferCopy{
                                           .srcOffset = sizeof(GPU_Car) * selectedCarId,
                                           .dstOffset = sizeof(GPU_Car) * selectedCarId, .size = sizeof(GPU_Car)
                                       });
                    }
                    if (inspectEdgeIdx >= 0 && inspectEdgeIdx < static_cast<std::int32_t>(totalEdges)) {
                        cmd.copyBuffer(*edgeBuffer, *edgeReadbackBuffer, vk::BufferCopy{
                                           .srcOffset = sizeof(GPU_Edge) * inspectEdgeIdx,
                                           .dstOffset = sizeof(GPU_Edge) * inspectEdgeIdx, .size = sizeof(GPU_Edge)
                                       });
                    }
                    if (inspectNodeIdx >= 0 && inspectNodeIdx < static_cast<std::int32_t>(cpuNodes.size())) {
                        cmd.copyBuffer(*nodeBuffer, *nodeReadbackBuffer, vk::BufferCopy{
                                           .srcOffset = sizeof(GPU_Node) * inspectNodeIdx,
                                           .dstOffset = sizeof(GPU_Node) * inspectNodeIdx, .size = sizeof(GPU_Node)
                                       });
                    }
                }
            }

            if (!isHeadless) {
                const vk::RenderPassBeginInfo renderPassInfo{
                    .renderPass = *renderPass, .framebuffer = *framebuffers[imageIndex],
                    .renderArea = {.offset = {0, 0}, .extent = swapchainExtent}, .clearValueCount = 1,
                    .pClearValues = &clearColor
                };
                cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

                const vk::Viewport dynamicViewport{
                    .x = 0.f, .y = 0.f, .width = static_cast<float>(swapchainExtent.width),
                    .height = static_cast<float>(swapchainExtent.height), .minDepth = 0.f, .maxDepth = 1.f
                };
                cmd.setViewport(0, dynamicViewport);
                const vk::Rect2D dynamicScissor{.offset = {0, 0}, .extent = swapchainExtent};
                cmd.setScissor(0, dynamicScissor);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *streetPipeline);

                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0,
                                       {*computeDescriptorSets[0]}, nullptr);
                mapBounds.selected_car_id = selectedCarId;
                cmd.pushConstants<GraphicsConstants>(*graphicsPipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0,
                                                     mapBounds);

                cmd.draw(2, totalEdges, 0, 0);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
                cmd.draw(6, totalCars, 0, 0);

                cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *exitNodePipeline);
                cmd.draw(6, totalEdges, 0, 0);

                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
                cmd.endRenderPass();
            }

            cmd.end();

            if (isHeadless) {
                queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd}, nullptr);
                queue.waitIdle();
                takeSnapshot();
            } else {
                constexpr vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
                const vk::SubmitInfo submitInfo{
                    .waitSemaphoreCount = 1, .pWaitSemaphores = &*imageAvailableSemaphore,
                    .pWaitDstStageMask = &waitStages, .commandBufferCount = 1, .pCommandBuffers = &*cmd,
                    .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore
                };
                queue.submit(submitInfo, nullptr);

                const vk::PresentInfoKHR presentInfo{
                    .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore, .swapchainCount = 1,
                    .pSwapchains = &*swapchain, .pImageIndices = &imageIndex
                };

                try {
                    auto presentResult = queue.presentKHR(presentInfo);
                    if (presentResult == vk::Result::eSuboptimalKHR || framebufferResized) {
                        framebufferResized = false;
                        recreateSwapchain();
                    }
                } catch (const vk::OutOfDateKHRError &) {
                    framebufferResized = false;
                    recreateSwapchain();
                }
                queue.waitIdle();
            }

            if (useGpu && hasTimestampQueries) {
                std::array<std::uint64_t, 8> timestamps{};
                const auto res = (*device).getQueryPoolResults(*timestampQueryPool, 0, 7, sizeof(timestamps),
                                                             timestamps.data(), sizeof(std::uint64_t),
                                                             vk::QueryResultFlagBits::e64);
                if (res == vk::Result::eSuccess) {
                    const float period = physicalDevice.getProperties().limits.timestampPeriod;
                    gpuClearMs = static_cast<double>(timestamps[1] - timestamps[0]) * period * 1e-6;
                    const double edge_barr_ms = static_cast<double>(timestamps[2] - timestamps[1]) * period * 1e-6;
                    gpuGridMs = static_cast<double>(timestamps[3] - timestamps[2]) * period * 1e-6;
                    const double car_barr_ms = static_cast<double>(timestamps[4] - timestamps[3]) * period * 1e-6;
                    gpuPhysMs = static_cast<double>(timestamps[5] - timestamps[4]) * period * 1e-6;
                    const double step_barr_ms = static_cast<double>(timestamps[6] - timestamps[5]) * period * 1e-6;
                    gpuBarrMs = edge_barr_ms + car_barr_ms + step_barr_ms;
                    gpuTotalMs = static_cast<double>(timestamps[6] - timestamps[0]) * period * 1e-6;

                    if (!printedGpuTimestamps && simTime >= 60.f) {
                        std::println("\n=======================================================");
                        std::println("=== REAL GPU HARDWARE TIMESTAMPS (Peak Active Load) ===");
                        std::println("=======================================================");
                        std::println("Edge Reset Pass (clear_edges.slang):      {:.4f} ms ({:.1f}%)", gpuClearMs, gpuClearMs / gpuTotalMs * 100.);
                        std::println("Spatial Grid Pass (build_grid.slang):     {:.4f} ms ({:.1f}%)", gpuGridMs, gpuGridMs / gpuTotalMs * 100.);
                        std::println("Agent IDM Physics Pass (physics.slang):   {:.4f} ms ({:.1f}%)", gpuPhysMs, gpuPhysMs / gpuTotalMs * 100.);
                        std::println("Pipeline Barriers & Memory Sync:          {:.4f} ms ({:.1f}%)", gpuBarrMs, gpuBarrMs / gpuTotalMs * 100.);
                        std::println("-------------------------------------------------------");
                        std::println("Total Compute Step Hardware Duration:     {:.4f} ms", gpuTotalMs);
                        std::println("=======================================================\n");
                        printedGpuTimestamps = true;
                    }
                }
            }
        }
        device.waitIdle();
        takeSnapshot();
        if (statEvacuated != lastEvacuatedCount || statRoad == 0) {
            recordPeriodicHistory();
        }
        saveResults();
    }

    void takeSnapshot() {
        if (useGpu) {
            const vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
            };
            const vk::raii::CommandBuffers cmdBuffers{device, allocInfo};
            const vk::raii::CommandBuffer &cmd = cmdBuffers.front();

            cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            const vk::BufferCopy carCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Car) * totalCars};
            cmd.copyBuffer(*carBuffer, *carReadbackBuffer, carCopy);

            const vk::BufferCopy edgeCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Edge) * totalEdges};
            cmd.copyBuffer(*edgeBuffer, *edgeReadbackBuffer, edgeCopy);

            const vk::BufferCopy nodeCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(GPU_Node) * cpuNodes.size()};
            cmd.copyBuffer(*nodeBuffer, *nodeReadbackBuffer, nodeCopy);

            cmd.end();

            queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd}, nullptr);
            queue.waitIdle();

            const void *mappedCars = carReadbackMemory.mapMemory(0, sizeof(GPU_Car) * totalCars);
            std::memcpy(cpuCars.data(), mappedCars, sizeof(GPU_Car) * totalCars);
            carReadbackMemory.unmapMemory();

            const void *mappedEdges = edgeReadbackMemory.mapMemory(0, sizeof(GPU_Edge) * totalEdges);
            std::memcpy(cpuEdges.data(), mappedEdges, sizeof(GPU_Edge) * totalEdges);
            edgeReadbackMemory.unmapMemory();

            const void *mappedNodes = nodeReadbackMemory.mapMemory(0, sizeof(GPU_Node) * cpuNodes.size());
            std::memcpy(cpuNodes.data(), mappedNodes, sizeof(GPU_Node) * cpuNodes.size());
            nodeReadbackMemory.unmapMemory();
        }

        statGarage = 0;
        statRoad = 0;
        statEvacuated = 0;
        statStuck = 0;
        statDisabled = 0;
        statAvgSpeed = 0.f;
        double totalSpeed = 0.;

        for (const auto &car: cpuCars) {
            switch (car.state) {
                case CarState::Driving:
                case CarState::Queuing:
                    ++statRoad;
                    totalSpeed += car.speed;
                    break;
                case CarState::Evacuated:
                    ++statEvacuated;
                    break;
                case CarState::Garage:
                    ++statGarage;
                    break;
                case CarState::Stuck:
                    ++statStuck;
                    break;
                case CarState::Disabled:
                    ++statDisabled;
                    break;
            }
        }
        statAvgSpeed = statRoad > 0 ? static_cast<float>(totalSpeed / statRoad) : 0.f;
    }

    [[nodiscard]] std::pair<float, float> screenToWorld(const double screenX, const double screenY) const {
        const float ndcX = mapBounds.aspect_ratio * (
                               static_cast<float>(screenX) / static_cast<float>(swapchainExtent.width) * 2.f - 1.f);
        const float ndcY = -(static_cast<float>(screenY) / static_cast<float>(swapchainExtent.height) * 2.f - 1.f);

        const float localX = ndcX * (mapBounds.extent_height * 0.5f) / mapBounds.zoom_level;
        const float localY = ndcY * (mapBounds.extent_height * 0.5f) / mapBounds.zoom_level;

        return {localX + mapBounds.camera_x, localY + mapBounds.camera_y};
    }

    [[nodiscard]] float worldDistanceToPixels(const float distanceMeters) const {
        return distanceMeters * mapBounds.zoom_level * static_cast<float>(swapchainExtent.height) / mapBounds.
               extent_height;
    }

    void selectClosestCar(const float worldX, const float worldY) {
        // no snapshot
        if (cpuCars.empty()) {
            return;
        }

        std::int32_t closestCarId = -1;
        float closestDist = std::numeric_limits<float>::max();

        for (std::size_t i = 0; i < cpuCars.size(); ++i) {
            const GPU_Car &car = cpuCars[i];

            if (car.state == CarState::Evacuated || car.state == CarState::Garage) {
                continue;
            }

            const GPU_Edge &edge = cpuEdges[car.current_edge_idx];
            const GPU_Node &startNode = cpuNodes[edge.start_node_idx];
            const GPU_Node &endNode = cpuNodes[edge.end_node_idx];

            const float t = edge.length > 0.f ? car.position / edge.length : 0.f;
            const float carWorldX = std::lerp(startNode.x, endNode.x, t);
            const float carWorldY = std::lerp(startNode.y, endNode.y, t);

            const float dx = carWorldX - worldX;
            const float dy = carWorldY - worldY;
            const float dist = std::hypot(dx, dy);

            if (dist < closestDist) {
                closestDist = dist;
                closestCarId = static_cast<std::int32_t>(i);
            }
        }

        constexpr float thresholdDegrees = 20.f / 111300.f; // 20 meters
        if (closestCarId != -1 && closestDist < thresholdDegrees) {
            selectedCarId = closestCarId;
            std::println("Selected Car ID: {}", selectedCarId);
        }
    }

    void applyMarqueeSelection() {
        if (cpuNodes.empty() || allExitNodes.empty()) {
            return;
        }

        const auto [startWorldX, startWorldY] = screenToWorld(startMouseX, startMouseY);
        const auto [currentWorldX, currentWorldY] = screenToWorld(currentMouseX, currentMouseY);

        const double dragDistX = std::abs(startMouseX - currentMouseX);
        const double dragDistY = std::abs(startMouseY - currentMouseY);

        if (dragDistX < 4. && dragDistY < 4.) {
            std::int32_t closestExitIdx = -1;
            float closestDist = std::numeric_limits<float>::max();

            for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
                const float ex = cpuNodes[allExitNodes[i]].x;
                const float ey = cpuNodes[allExitNodes[i]].y;
                const float dx = ex - startWorldX;
                const float dy = ey - startWorldY;
                const float dist = std::hypot(dx, dy);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestExitIdx = static_cast<std::int32_t>(i);
                }
            }

            constexpr float thresholdDegrees = 50.f / 111300.f; // 50 meters
            if (closestExitIdx != -1 && closestDist < thresholdDegrees) {
                isExitOpen[closestExitIdx] = !isExitOpen[closestExitIdx];
                triggerDynamicReroute();
            }
            return;
        }

        const float minX = std::min(startWorldX, currentWorldX);
        const float maxX = std::max(startWorldX, currentWorldX);
        const float minY = std::min(startWorldY, currentWorldY);
        const float maxY = std::max(startWorldY, currentWorldY);

        bool changed = false;
        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            const float x = cpuNodes[allExitNodes[i]].x;
            const float y = cpuNodes[allExitNodes[i]].y;

            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
                isExitOpen[i] = !isExitOpen[i];
                changed = true;
            }
        }

        if (changed) {
            triggerDynamicReroute();
        }
    }

    void recordPeriodicHistory() {
        const std::int32_t flow = statEvacuated - lastEvacuatedCount; // cars per minute
        flowrateHistory.push_back(flow);
        evacuatedHistory.push_back(statEvacuated);
        lastEvacuatedCount = statEvacuated;

        garageHistory.push_back(statGarage);
        roadHistory.push_back(statRoad);
        stuckHistory.push_back(statStuck);
        avgSpeedHistory.push_back(statAvgSpeed * 3.6f);

        std::vector currentEvacCount(allExitNodes.size(), 0);
        for (const auto &car: cpuCars) {
            if (car.state == CarState::Evacuated) {
                if (car.current_edge_idx >= 0 && car.current_edge_idx < static_cast<std::int32_t>(totalEdges)) {
                    std::int32_t endNode = cpuEdges[car.current_edge_idx].end_node_idx;
                    if (auto it = std::ranges::find(allExitNodes, endNode); it != allExitNodes.end()) {
                        const std::size_t exitIdx = std::distance(allExitNodes.begin(), it);
                        ++currentEvacCount[exitIdx];
                    }
                }
            }
        }

        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            std::int32_t exitFlow = currentEvacCount[i] - exitLastEvacuatedCount[i];
            exitFlowrateHistory[i].push_back(exitFlow);
            exitLastEvacuatedCount[i] = currentEvacCount[i];
        }

        lastRecordTime = simTime;
    }

    void saveResults() {
        nlohmann::json results;
        results["use_gpu"] = useGpu;
        if (!useGpu && cpuEngine) {
            results["cpu_threads"] = cpuEngine->getNumThreads();
        }
        results["vulkan_nodes_path"] = vulkanNodesPath;
        results["vulkan_edges_path"] = vulkanEdgesPath;
        results["vulkan_cars_path"] = vulkanCarsPath;
        results["participation"] = participation;
        results["total_cars"] = totalCars;
        results["simulation_time_seconds"] = simTime;

        const auto benchmarkEndTime = std::chrono::high_resolution_clock::now();
        const double wallTimeSec = std::chrono::duration<double>(benchmarkEndTime - benchmarkStartTime).count();
        const double totalSteps = static_cast<double>(simTime) * 10.;
        results["benchmark"]["wall_clock_time_seconds"] = wallTimeSec;
        results["benchmark"]["simulated_time_seconds"] = simTime;
        results["benchmark"]["real_time_factor"] = wallTimeSec > 0.
                                                       ? static_cast<double>(simTime) / wallTimeSec
                                                       : 0.;
        results["benchmark"]["steps_simulated"] = static_cast<std::int64_t>(totalSteps);
        results["benchmark"]["mean_step_time_ms"] = totalSteps > 0. && wallTimeSec > 0.
                                                        ? wallTimeSec * 1000. / totalSteps
                                                        : 0.;


        results["history"]["flowrate_cars_per_min"] = flowrateHistory;
        results["history"]["evacuated_cumulative"] = evacuatedHistory;
        results["history"]["waiting_in_garage"] = garageHistory;
        results["history"]["active_on_road"] = roadHistory;
        results["history"]["stuck_cars"] = stuckHistory;
        results["history"]["city_average_speed_kmh"] = avgSpeedHistory;

        nlohmann::json exitsJson = nlohmann::json::array();
        for (std::size_t i = 0; i < allExitNodes.size(); ++i) {
            nlohmann::json exitItem;
            exitItem["exit_index"] = i;
            exitItem["node_id"] = allExitNodes[i];
            exitItem["is_open"] = isExitOpen[i];
            exitItem["flowrate_history_cars_per_min"] = exitFlowrateHistory[i];
            exitsJson.push_back(exitItem);
        }
        results["exits"] = exitsJson;

        if (std::ofstream out{outputFilePath}; out.is_open()) {
            out << results.dump(4);
            std::println("Simulation results saved to {}", outputFilePath);
        } else {
            std::println("Error: Could not open output file {} to write results.", outputFilePath);
        }
    }
};

std::int32_t main(const std::int32_t argc, const char *argv[]) {
    std::string configPath = "config.json";
    if (argc > 1) {
        configPath = argv[1];
    }
    try {
        EvacuationEngine engine{};
        engine.loadConfig(configPath);
        engine.run();
    } catch (const std::exception &e) {
        std::println(std::cerr, "Fatal GPU Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

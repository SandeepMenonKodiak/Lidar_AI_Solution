#include <NvInfer.h>
#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

using namespace nvinfer1;

constexpr size_t INPUT_SIZE = 1024*1024;  // <-- Define INPUT_SIZE; update as needed

// Logger for TensorRT info/warning/errors
class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // remove this 'if' if you want to see warnings as well
        if ((severity == Severity::kINFO) || (severity == Severity::kWARNING)) return;
        std::cerr << msg << std::endl;
    }
} gLogger;

int main()
{
    void* handle = dlopen("/home/Lidar_AI_Solution/CUDA-BEVFusion/plugin_codes/GridsampleIPluginV2DynamicExt/gridSamplerPlugin.so", RTLD_LAZY);
    if (!handle) {
        // Handle error - the .so file cannot be loaded
        std::cerr << "Cannot load library: " << dlerror() << '\n';
        return 1;
    }

    // Create a TensorRT engine.
    const std::string engineFile = "/home/Lidar_AI_Solution/CUDA-BEVFusion/model/segm/build/head.map.plan";
    
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineFile << " error!" << std::endl;
        return -1;
    }

    // Load engine file
    file.seekg(0, file.end);
    long int fsize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(engine != nullptr);

    // Create context
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Prepare input data
    // Create GPU buffers on device
    void *data;
    cudaMalloc(&data, INPUT_SIZE * sizeof(float));

    // TODO: Copy your data to the 'data' pointer here

    // Bind input data
    void* buffers[] = { data };
    
    // Execute model on device
    context->executeV2(buffers);

    // Release the stream and the buffers
    cudaFree(data);

    // Release context and engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    dlclose(handle);

    return 0;
}

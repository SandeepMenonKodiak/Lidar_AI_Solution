#include <cuda_runtime.h>
#include <string.h>
#include <cuda_fp16.h>
#include <spconv/engine.hpp>

#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"
#include <dlfcn.h>
#include <cfloat>
#include <limits>
#include <iostream>


using Color = std::tuple<unsigned char, unsigned char, unsigned char>;

// Define the MAP_PALETTE
std::map<std::string, Color> MAP_PALETTE = {
    {"drivable_area", {166, 206, 227}},
    {"road_segment", {31, 120, 180}},
    {"road_block", {178, 223, 138}},
    {"lane", {51, 160, 44}},
    {"ped_crossing", {251, 154, 153}},
    {"walkway", {227, 26, 28}},
    {"stop_line", {253, 191, 111}},
    {"carpark_area", {255, 127, 0}},
    {"road_divider", {202, 178, 214}},
    {"lane_divider", {106, 61, 154}},
    {"divider", {106, 61, 154}}
};

std::vector<std::string> map_classes = {
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider"
};

void visualize_map(const std::string& fpath, const bevfusion::head::mapsegm::CanvasOutput& masks, const std::vector<std::string>& classes) {
    int batch = masks.size();
    if (batch == 0) return;  // no masks to visualize

    int channels = masks[0].size();
    if (channels == 0) return;

    int height = masks[0][0].size();
    if (height == 0) return;

    int width = masks[0][0][0].size();
    if (width == 0) return;
    printf("width is: %d, height is: %d, channels is: %d, batch_size is: %d\n", width, height, channels, batch);
    
    // Create an empty image with 3 channels (RGB)
    std::vector<unsigned char> image(height * width * 3, 240);  // assuming the default background is (240,240,240)
    std::map<float, int> frequency; // Key: number, Value: count
    
    // Here, let's assume that you're only interested in visualizing the first batch of masks.
    for (size_t k = 0; k < classes.size() && k < channels; ++k) {
        float max_value = FLT_MIN;
        float min_value = FLT_MAX;
        const std::string& name = classes[k];
        if (MAP_PALETTE.find(name) != MAP_PALETTE.end()) {
            Color color = MAP_PALETTE[name];
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                  // frequency[masks[0][k][i][j]]++;
                    max_value = std::max(max_value, masks[0][k][i][j]);
                    min_value = std::min(min_value, masks[0][k][i][j]);
                    if (masks[0][k][i][j] >= 0.5) {
                        int index = (i * width + j) * 3;
                        // printf("color: %d %d %d\n", std::get<0>(color), std::get<1>(color), std::get<2>(color));
                        image[index] = std::get<0>(color);
                        image[index + 1] = std::get<1>(color);
                        image[index + 2] = std::get<2>(color);
                    }

                }
            }
        }
      printf("[Channel: %s] max_value is: %f, min_value is: %f\n", name.c_str(), max_value, min_value);

    }
    

    stbi_write_jpg(fpath.c_str(), width, height, 3, &image[0], 100);
}

bool test_head(const std::string& model, const std::string& precision, const std::string& data) {
    
    try {
      // Dummy data
      auto input_data = nv::Tensor::load(nv::format("%s/head_sig_input.tensor", data.c_str()), false);
      const nvtype::half* input_tensor_ptr = input_data.ptr<nvtype::half>();

      // Load model
      bevfusion::head::mapsegm::MapSegHeadParameter mapsegm;
      mapsegm.model = nv::format("model/%s/build/head_sig.map.plan", model.c_str());
      std::shared_ptr<bevfusion::head::mapsegm::MapSegHead> mapsegm_;
      mapsegm_ = bevfusion::head::mapsegm::create_mapseghead(mapsegm);
      std::vector<int> input_shape = mapsegm_->input_shape();

      //Forward pass
      cudaStream_t _stream;
      cudaStreamCreate(&_stream);
      bevfusion::head::mapsegm::CanvasOutput maps = mapsegm_->forward(input_tensor_ptr, _stream);
      checkRuntime(cudaStreamSynchronize(_stream));
      
      // Save output
      visualize_map("build/cuda-cameramapsegmtest.jpg", maps, map_classes);

      cudaStreamDestroy(_stream);

      return true;
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return false;
    }
}

bool load_grid_sampler_plugin() {
  void* handle = dlopen("/home/Lidar_AI_Solution/CUDA-BEVFusion/plugin_codes/GridsampleIPluginV2DynamicExt/gridSamplerPlugin.so", RTLD_LAZY);
    if (!handle) {
        // Handle error - the .so file cannot be loaded
        printf("Cannot load library: %s\n", dlerror());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {

  if(!load_grid_sampler_plugin()){
    printf("Cannot load grid sampler plugin.\n");
    return -1;
  }

  const char* data      = "exampletest";
  const char* model     = "cameramapsegm";
  const char* precision = "fp16";

  if (argc > 1) data      = argv[1];
  if (argc > 2) model     = argv[2];
  if (argc > 3) precision = argv[3];

  printf("Data: %s\n", data);
  printf("Model: %s\n", model);
  printf("Precision: %s\n", precision);

  if (!test_head(model, precision, data)) {
      printf("Test failed.\n");
      return -1;
  }

}

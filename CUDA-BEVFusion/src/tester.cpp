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


bevfusion::head::mapsegm::CanvasOutput copyToCanvasOutput(const nvtype::half* data, const std::vector<int64_t>& shape) {
    bevfusion::head::mapsegm::CanvasOutput output;
    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    half* out_data = nullptr;
    checkRuntime(cudaMallocHost(&out_data, batch_size * channels * height * width * sizeof(half)));
    checkRuntime(cudaMemcpy(out_data, data, batch_size * channels * height * width * sizeof(half), cudaMemcpyDeviceToHost));

    output.resize(batch_size);
    for(int b = 0; b<batch_size;++b){
        output[b].resize(channels);
        for(int c = 0; c<channels;++c){
            output[b][c].resize(height);
            for(int h = 0; h<height;++h){
                output[b][c][h].resize(width);
                for(int w = 0; w<width;++w){
                    int idx = b*channels*height*width + c*height*width + h*width + w;
                    output[b][c][h][w] = __half2float(out_data[idx]);
                }
            }
        }
    }

    if(out_data)
        checkRuntime(cudaFreeHost(out_data));

    return output;
}

bevfusion::head::mapregr::RegrOutput copyToRegrOutput(const nvtype::half* data, const std::vector<int64_t>& shape) {
    bevfusion::head::mapregr::RegrOutput output;
    int batch_size = shape[0];
    int height = shape[1];
    int width = shape[2];

    half* out_data = nullptr;
    checkRuntime(cudaMallocHost(&out_data, batch_size * height * width * sizeof(half)));
    checkRuntime(cudaMemcpy(out_data, data, batch_size * height * width * sizeof(half), cudaMemcpyDeviceToHost));

    output.resize(batch_size);
    for(int b = 0; b<batch_size;++b){
            output[b].resize(height);
            for(int h = 0; h<height;++h){
                output[b][h].resize(width);
                for(int w = 0; w<width;++w){
                    int idx = b*height*width + h*width + w;
                    output[b][h][w] = __half2float(out_data[idx]);
                }
            }
        }
    

    if(out_data)
        checkRuntime(cudaFreeHost(out_data));

    return output;
}

bool areCanvasOutputsClose(const bevfusion::head::mapsegm::CanvasOutput v1, const bevfusion::head::mapsegm::CanvasOutput v2, float epsilon) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i].size() != v2[i].size()) return false;
        for (size_t j = 0; j < v1[i].size(); ++j) {
            if (v1[i][j].size() != v2[i][j].size()) return false;
            for (size_t k = 0; k < v1[i][j].size(); ++k) {
                if (v1[i][j][k].size() != v2[i][j][k].size()) return false;
                for (size_t l = 0; l < v1[i][j][k].size(); ++l) {
                    if (std::abs(v1[i][j][k][l] - v2[i][j][k][l]) > epsilon) {
                      std::cout << "v1[" << i << "][" << j << "][" << k << "][" << l << "] = " << v1[i][j][k][l] << std::endl;
                      std::cout << "v2[" << i << "][" << j << "][" << k << "][" << l << "] = " << v2[i][j][k][l] << std::endl;
                      return false;
                    }
                }
            }
        }
    }
    return true;
}

bool areRegrOutputsClose(const bevfusion::head::mapregr::RegrOutput v1, const bevfusion::head::mapregr::RegrOutput v2, float epsilon) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
      if (v1[i].size() != v2[i].size()) return false;
      for (size_t j = 0; j < v1[i].size(); ++j) {
          if (v1[i][j].size() != v2[i][j].size()) return false;
          for (size_t k = 0; k < v1[i][j].size(); ++k) {
            if (std::abs(v1[i][j][k] - v2[i][j][k]) > epsilon) {
              return false;
            }
          }
        }
      }
    return true;
}


bool test_seghead(const std::string& model, const std::string& precision, const std::string& data) {
    
    try {
      // Dummy data
      auto input_data = nv::Tensor::load(nv::format("%s/head_sig_input.tensor", data.c_str()), false);
      auto output = nv::Tensor::load(nv::format("%s/head_sig_output.tensor", data.c_str()), false);
      const nvtype::half* input_tensor_ptr = input_data.ptr<nvtype::half>();

      // Load model
      bevfusion::head::mapsegm::MapSegHeadParameter mapsegm;
      mapsegm.model = nv::format("model/%s/build/head_sig.map.plan", model.c_str());
      std::shared_ptr<bevfusion::head::mapsegm::MapSegHead> mapsegm_ = bevfusion::head::mapsegm::create_mapseghead(mapsegm);
      std::vector<int> input_shape = mapsegm_->input_shape();

      //Forward pass
      cudaStream_t _stream;
      cudaStreamCreate(&_stream);
      bevfusion::head::mapsegm::CanvasOutput maps = mapsegm_->forward(input_tensor_ptr, _stream);
      checkRuntime(cudaStreamSynchronize(_stream));
      
      // Save output
      visualize_map("build/cuda-cameramapsegmtest.jpg", maps, map_classes);

      cudaStreamDestroy(_stream);

      bevfusion::head::mapsegm::CanvasOutput true_maps = copyToCanvasOutput(output.ptr<nvtype::half>(), output.shape);

      return areCanvasOutputsClose(maps, true_maps, 1e-1);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return false;
    }
}

bool test_camsegdecoder(const std::string& model, const std::string& precision, const std::string& data) {
  try{
    // Dummy data
    auto input_data = nv::Tensor::load(nv::format("%s/camera_bev_features.tensor", data.c_str()), false);
    auto output_data = nv::Tensor::load(nv::format("%s/head_sig_input.tensor", data.c_str()), false);
    std::vector<int64_t> input_shape = input_data.shape;
    std::vector<int64_t> output_shape = output_data.shape;
    for(int i = 0; i < input_shape.size(); i++){
      printf("input_shape[%d] is: %ld\n", i, input_shape[i]);
    }
    for(int i = 0; i < output_shape.size(); i++){
      printf("output_shape[%d] is: %ld\n", i, output_shape[i]);
    }
    const nvtype::half* camera_bevfeat = input_data.ptr<nvtype::half>();

    // Load model
    const std::string cameradecoder_param = nv::format("model/%s/build/camera_decoder.plan", model.c_str());
    std::shared_ptr<bevfusion::fuser::CameraDecoder> cameradecoder_ = bevfusion::fuser::create_cameradecoder(cameradecoder_param);

    //Forward pass
    cudaStream_t _stream;
    cudaStreamCreate(&_stream);

    const nvtype::half* cameradecoder_feature = cameradecoder_->forward(camera_bevfeat, _stream);
    checkRuntime(cudaStreamSynchronize(_stream));
    cudaStreamDestroy(_stream);

    // Compare output
    bevfusion::head::mapsegm::CanvasOutput model_out = copyToCanvasOutput(cameradecoder_feature, output_shape);
    bevfusion::head::mapsegm::CanvasOutput output = copyToCanvasOutput(output_data.ptr<nvtype::half>(), output_shape);
    int batch_size = output_shape[0];
    int channels = output_shape[1];
    int height = output_shape[2];
    int width = output_shape[3];

    float pred_max_value = FLT_MIN;
    float out_max_value = FLT_MIN;
    for(int b = 0; b<batch_size;++b){
      for(int c = 0; c<channels;++c){
        for(int h = 0; h<height;++h){
          for(int w = 0; w<width;++w){
            pred_max_value = std::max(pred_max_value, model_out[b][c][h][w]);
            out_max_value = std::max(out_max_value, output[b][c][h][w]);
          }
        }
      }
    }

    printf("pred_max_value is: %f, out_max_value is: %f\n", pred_max_value, out_max_value);

    return areCanvasOutputsClose(model_out, output, 1e-1);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }

}

bool test_lidarsegdecoder(const std::string& model, const std::string& precision, const std::string& data) {
  try{
    // Dummy data
    auto input_data = nv::Tensor::load(nv::format("%s/lidar_features.tensor", data.c_str()), false);
    auto output_data = nv::Tensor::load(nv::format("%s/head_sig_input.tensor", data.c_str()), false);
    std::vector<int64_t> input_shape = input_data.shape;
    std::vector<int64_t> output_shape = output_data.shape;
    for(int i = 0; i < input_shape.size(); i++){
      printf("input_shape[%d] is: %ld\n", i, input_shape[i]);
    }
    for(int i = 0; i < output_shape.size(); i++){
      printf("output_shape[%d] is: %ld\n", i, output_shape[i]);
    }
    const nvtype::half* lidar_bevfeat = input_data.ptr<nvtype::half>();

    // Load model
    const std::string lidardecoder_param = nv::format("model/%s/build/lidar_decoder.plan", model.c_str());
    std::shared_ptr<bevfusion::fuser::LidarDecoder> lidardecoder_ = bevfusion::fuser::create_lidardecoder(lidardecoder_param);

    //Forward pass
    cudaStream_t _stream;
    cudaStreamCreate(&_stream);

    const nvtype::half* lidardecoder_feature = lidardecoder_->forward(lidar_bevfeat, _stream);
    checkRuntime(cudaStreamSynchronize(_stream));
    cudaStreamDestroy(_stream);

    // Compare output
    bevfusion::head::mapsegm::CanvasOutput model_out = copyToCanvasOutput(lidardecoder_feature, output_shape);
    bevfusion::head::mapsegm::CanvasOutput output = copyToCanvasOutput(output_data.ptr<nvtype::half>(), output_shape);
    int batch_size = output_shape[0];
    int channels = output_shape[1];
    int height = output_shape[2];
    int width = output_shape[3];

    float pred_max_value = FLT_MIN;
    float out_max_value = FLT_MIN;
    for(int b = 0; b<batch_size;++b){
      for(int c = 0; c<channels;++c){
        for(int h = 0; h<height;++h){
          for(int w = 0; w<width;++w){
            pred_max_value = std::max(pred_max_value, model_out[b][c][h][w]);
            out_max_value = std::max(out_max_value, output[b][c][h][w]);
          }
        }
      }
    }
    printf("pred_max_value is: %f, out_max_value is: %f\n", pred_max_value, out_max_value);
    return areCanvasOutputsClose(model_out, output, 1e-1);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }

}

bool load_grid_sampler_plugin() {
  void* handle = dlopen("/home/Lidar_AI_Solution/CUDA-BEVFusion/trt_plugins/GridsampleIPluginV2DynamicExt/libGridSamplerPlugin.so", RTLD_LAZY);
    if (!handle) {
        // Handle error - the .so file cannot be loaded
        printf("Cannot load library: %s\n", dlerror());
        return false;
    }
    return true;
}

bool test_lidarsegencoder(const std::string& model, const std::string& precision, const std::string& data) {
    try {
      // Dummy data
      auto input_data = nv::Tensor::load(nv::format("%s/points.tensor", data.c_str()), false);
      auto output_data = nv::Tensor::load(nv::format("%s/lidar_features.tensor", data.c_str()), false);
      std::vector<int64_t> input_shape = input_data.shape;
      std::vector<int64_t> output_shape = output_data.shape;
      int num_points = input_data.size(0);
      nvtype::half* lidar_points = input_data.ptr<nvtype::half>();

      for(int i = 0; i < input_shape.size(); i++){
        printf("input_shape[%d] is: %ld\n", i, input_shape[i]);
      }
      for(int i = 0; i < output_shape.size(); i++){
        printf("output_shape[%d] is: %ld\n", i, output_shape[i]);
      }

      // Load model
      bevfusion::lidar::VoxelizationParameter voxelization;
      voxelization.min_range = nvtype::Float3(-51.2f, -51.2f, -5.0);
      voxelization.max_range = nvtype::Float3(+51.2f, +51.2f, +3.0);
      voxelization.voxel_size = nvtype::Float3(0.1f, 0.1f, 0.2f);
      voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
      voxelization.max_points_per_voxel = 10;
      voxelization.max_points = 300000;
      voxelization.max_voxels = 120000;
      voxelization.num_feature = 5;
      bevfusion::lidar::SCNParameter lidar_scn;
      lidar_scn.voxelization = voxelization;
      lidar_scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
      lidar_scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

      nvtype::half* lidar_points_device_ = nullptr;
      nvtype::half* lidar_points_host_ = nullptr;
      size_t capacity_points_ = 300000;
      size_t bytes_capacity_points_ = capacity_points_ * lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
      std::shared_ptr<bevfusion::lidar::SCN> lidar_scn_ = bevfusion::lidar::create_scn(lidar_scn);
      checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
      checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));  // Pinned buffer. Accessible from GPU.

      //Forward pass
      cudaStream_t _stream;
      cudaStreamCreate(&_stream);

      size_t bytes_points = num_points * lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
      checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
      checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

      const nvtype::half* lidar_feature = lidar_scn_->forward(lidar_points_device_, num_points, _stream);
      checkRuntime(cudaStreamSynchronize(_stream));

      checkRuntime(cudaStreamSynchronize(_stream));
      cudaStreamDestroy(_stream);

      if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
      if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));

      // Compare output
      bevfusion::head::mapsegm::CanvasOutput model_out = copyToCanvasOutput(lidar_feature, output_shape);
      bevfusion::head::mapsegm::CanvasOutput output = copyToCanvasOutput(output_data.ptr<nvtype::half>(), output_shape);

      int batch_size = output_shape[0];
      int channels = output_shape[1];
      int height = output_shape[2];
      int width = output_shape[3];

      float pred_max_value = FLT_MIN;
      float out_max_value = FLT_MIN;
      for(int b = 0; b<batch_size;++b){
        for(int c = 0; c<channels;++c){
          for(int h = 0; h<height;++h){
            for(int w = 0; w<width;++w){
              pred_max_value = std::max(pred_max_value, model_out[b][c][h][w]);
              out_max_value = std::max(out_max_value, output[b][c][h][w]);
            }
          }
        }
      }
      printf("pred_max_value is: %f, out_max_value is: %f\n", pred_max_value, out_max_value);

      return areCanvasOutputsClose(model_out, output, 0.5);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return false;
    }
}

bool test_regrmodel(const std::string& model, const std::string& precision, const std::string& data) {
  try{
    // Dummy data
    auto input_data = nv::Tensor::load(nv::format("%s/points.tensor", data.c_str()), false);
    auto output_data = nv::Tensor::load(nv::format("%s/prediction.tensor", data.c_str()), false);
    std::vector<int64_t> input_shape = input_data.shape;
    std::vector<int64_t> output_shape = output_data.shape;
    int num_points = input_data.size(0);
    nvtype::half* lidar_points = input_data.ptr<nvtype::half>();

    for(int i = 0; i < input_shape.size(); i++){
      printf("input_shape[%d] is: %ld\n", i, input_shape[i]);
    }
    for(int i = 0; i < output_shape.size(); i++){
      printf("output_shape[%d] is: %ld\n", i, output_shape[i]);
    }

    // Load model
    bevfusion::lidar::VoxelizationParameter voxelization;
    voxelization.min_range = nvtype::Float3(-51.2f, -51.2f, -5.0);
    voxelization.max_range = nvtype::Float3(+51.2f, +51.2f, +3.0);
    voxelization.voxel_size = nvtype::Float3(0.1f, 0.1f, 0.2f);
    voxelization.grid_size =
    voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
    voxelization.max_points_per_voxel = 10;
    voxelization.max_points = 300000;
    voxelization.max_voxels = 120000;
    voxelization.num_feature = 5;
    bevfusion::lidar::SCNParameter lidar_scn;
    lidar_scn.voxelization = voxelization;
    lidar_scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
    lidar_scn.order = bevfusion::lidar::CoordinateOrder::XYZ;
    bevfusion::head::mapregr::MapRegrHeadParameter mapregr_param;
    mapregr_param.model = nv::format("model/%s/build/head.map.plan", model.c_str());
    
    nvtype::half* lidar_points_device_ = nullptr;
    nvtype::half* lidar_points_host_ = nullptr;
    size_t capacity_points_ = 300000;
    size_t bytes_capacity_points_ = capacity_points_ * lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    std::shared_ptr<bevfusion::lidar::SCN> lidar_scn_ = bevfusion::lidar::create_scn(lidar_scn);
    const std::string lidardecoder_param = nv::format("model/%s/build/lidar_decoder.plan", model.c_str());
    std::shared_ptr<bevfusion::fuser::LidarDecoder> lidardecoder_ = bevfusion::fuser::create_lidardecoder(lidardecoder_param);
    std::shared_ptr<bevfusion::head::mapregr::MapRegrHead> mapregr_ = bevfusion::head::mapregr::create_mapregrhead(mapregr_param);

    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));

    //Forward pass
    cudaStream_t _stream;
    cudaStreamCreate(&_stream);

    size_t bytes_points = num_points * lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

    const nvtype::half* lidar_feature = lidar_scn_->forward(lidar_points_device_, num_points, _stream);
    const nvtype::half* lidardecoder_feature = lidardecoder_->forward(lidar_feature, _stream);
    const bevfusion::head::mapregr::RegrOutput model_out = mapregr_->forward(lidardecoder_feature, _stream);
    
    checkRuntime(cudaStreamSynchronize(_stream));
    checkRuntime(cudaStreamSynchronize(_stream));
    cudaStreamDestroy(_stream);
    
    const bevfusion::head::mapregr::RegrOutput output = copyToRegrOutput(output_data.ptr<nvtype::half>(), output_shape);

     int batch_size = output_shape[0];
      int height = output_shape[1];
      int width = output_shape[2];

      float pred_max_value = FLT_MIN;
      float out_max_value = FLT_MIN;
      float pred_min_value = FLT_MAX;
      float out_min_value = FLT_MAX;
      for(int b = 0; b<batch_size;++b){
          for(int h = 0; h<height;++h){
            for(int w = 0; w<width;++w){
              pred_max_value = std::max(pred_max_value, model_out[b][h][w]);
              out_max_value = std::max(out_max_value, output[b][h][w]);
              pred_min_value = std::min(pred_min_value, model_out[b][h][w]);
              out_min_value = std::min(out_min_value, output[b][h][w]);
            }
          }
        }
      
      printf("pred_max_value is: %f, out_max_value is: %f\n", pred_max_value, out_max_value);
      printf("pred_min_value is: %f, out_min_value is: %f\n", pred_min_value, out_min_value);

    return areRegrOutputsClose(model_out, output, 1e-1);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }

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

  if(strstr(model, "mapregr") != NULL) {
    if(!test_regrmodel(model, precision, data)) {
      printf("[test_regrmodel] Test failed.\n");
      return -1;
    }
  }

  if(strstr(model, "mapsegm") != NULL) {
    if (!test_seghead(model, precision, data)) {
        printf("[test_seghead] Test failed.\n");
        return -1;
    }
  }

  if(strcmp(model, "cameramapsegm") == 0) {
    if (!test_camsegdecoder(model, precision, data)) {
        printf("[test_camsegdecoder] Test failed.\n");
        return -1;
    }
  }

  if(strcmp(model, "lidarmapsegm") == 0) {
    if (!test_lidarsegencoder(model, precision, data)) {
      printf("[test_lidarsegencoder] Test failed.\n");
      return -1;
    }

    if (!test_lidarsegdecoder(model, precision, data)) {
      printf("[test_lidarsegdecoder] Test failed.\n");
      return -1;
    }
  }
}

#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-mapsegm.hpp"
#include <iostream>
#include <map>
#include <cfloat>
#include <limits>

namespace bevfusion {
namespace head {
namespace mapsegm {

class MapSegHeadImplement : public MapSegHead {
 public:
  virtual ~MapSegHeadImplement() {
    if (out_bindings_) checkRuntime(cudaFree(out_bindings_));
    if (output_host_map_) checkRuntime(cudaFreeHost(output_host_map_));
  }

  virtual bool init(const MapSegHeadParameter& param) {
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }
    
    printf("MapSegHeadImplement::init\n");

    param_ = param;
    create_binding_memory();
    printf("Input binding shape is: %d %d %d %d\n", inp_bindshape_[0], inp_bindshape_[1], inp_bindshape_[2], inp_bindshape_[3]);
    std::cout << "Output Binding shape is: " << out_bindshape_[0] << " " << out_bindshape_[1] << " " << out_bindshape_[2] << " "
              << out_bindshape_[3] << std::endl;
    checkRuntime(cudaMallocHost(&output_host_map_, out_bindshape_[0] * out_bindshape_[1] * out_bindshape_[2] * out_bindshape_[3] * sizeof(half)));
    checkRuntime(cudaMallocHost(&input_features_, inp_bindshape_[0] * inp_bindshape_[1] * inp_bindshape_[2] * inp_bindshape_[3] * sizeof(half)));
    return true;
  }

  void create_binding_memory() {
    std::cout<<"create binding memory"<<std::endl;
    std::cout<<"engine_->num_bindings() is: "<<engine_->num_bindings()<<std::endl;
    inp_bindshape_ = engine_->static_dims(0);
    out_bindshape_ = engine_->static_dims(1);
    size_t volumn = std::accumulate(out_bindshape_.begin(), out_bindshape_.end(), 1, std::multiplies<int>());

    checkRuntime(cudaMalloc(&out_bindings_, volumn * sizeof(half)));
  }

  virtual void print() override { engine_->print("MapSegHead"); }

  virtual CanvasOutput forward(const nvtype::half* transfusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward({/* input  */ transfusion_feature,
                      /* output */ out_bindings_}, _stream);

    checkRuntime(cudaStreamSynchronize(_stream)); // wait for device to finish the copy
    checkRuntime(cudaMemcpyAsync(input_features_, transfusion_feature, inp_bindshape_[0] * inp_bindshape_[1] * inp_bindshape_[2] * inp_bindshape_[3] * sizeof(half), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaMemcpyAsync(output_host_map_, out_bindings_, out_bindshape_[0] * out_bindshape_[1] * out_bindshape_[2] * out_bindshape_[3] * sizeof(half), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream)); // wait for device to finish the copy

    int input_width = inp_bindshape_[3];
    int input_height = inp_bindshape_[2];
    int input_channels = inp_bindshape_[1];
    int input_batch_size = inp_bindshape_[0];
    printf("input_width is: %d, input_height is: %d, input_channels is: %d, input_batch_size is: %d\n", input_width, input_height, input_channels, input_batch_size);
    std::map<float, int> inp_frequency; // Key: number, Value: count
    float max_value = FLT_MIN;
    float min_value = FLT_MAX;
    for(int b = 0; b < input_batch_size; b++) {
        for(int c = 0; c < input_channels; c++) {
            for(int h = 0; h < input_height; h++) {
                for(int w = 0; w < input_width; w++) {
                    int idx = b * (input_channels * input_height * input_width) + c * (input_height * input_width) + h * input_width + w;
                    float value = __half2float(input_features_[idx]);
                    inp_frequency[value]++;
                    if(value > max_value) {
                        max_value = value;
                    }
                    if(value < min_value) {
                        min_value = value;
                    }
                }
            }
        }
    }
    printf("[Neck] max_value is: %f, min_value is: %f\n", max_value, min_value);
    // printf("Input frequency is: \n");
    // for(auto it = inp_frequency.begin(); it != inp_frequency.end(); it++) {
    //     std::cout << it->first << " " << it->second << std::endl;
    // }

    CanvasOutput output;
    int width = out_bindshape_[3];
    int height = out_bindshape_[2];
    int channels = out_bindshape_[1];
    int batch_size = out_bindshape_[0];
    output.resize(batch_size);
    max_value = FLT_MIN;
    min_value = FLT_MAX;
    printf("width is: %d, height is: %d, channels is: %d, batch_size is: %d\n", width, height, channels, batch_size);
    std::map<float, int> frequency; // Key: number, Value: count
    for (int b = 0; b < batch_size; b++) {
        output[b].resize(channels);
        for (int c = 0; c < channels; c++) {
            output[b][c].resize(height);
            for (int h = 0; h < height; h++) {
                output[b][c][h].resize(width);
                for (int w = 0; w < width; w++) {
                    int idx = b * (channels * height * width) + c * (height * width) + h * width + w;
                    output[b][c][h][w] = __half2float(output_host_map_[idx]);
                    frequency[output[b][c][h][w]]++;
                    if(output[b][c][h][w] > max_value) {
                        max_value = output[b][c][h][w];
                    }
                    if(output[b][c][h][w] < min_value) {
                        min_value = output[b][c][h][w];
                    }
                }
            }
        }
    }

    printf("[Head] max_value is: %f, min_value is: %f\n", max_value, min_value);

    // for(const auto &pair : frequency) {
    //     std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    // }

    return output;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  half* out_bindings_;
  std::vector<int> out_bindshape_;
  std::vector<int> inp_bindshape_;
  MapSegHeadParameter param_;
  half* output_host_map_ = nullptr;
  half* input_features_ = nullptr;

};

std::shared_ptr<MapSegHead> create_mapseghead(const MapSegHeadParameter& param) {
  std::shared_ptr<MapSegHeadImplement> instance(new MapSegHeadImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace mapsegm
};  // namespace head
};  // namespace bevfusion

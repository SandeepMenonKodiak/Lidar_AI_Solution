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
  virtual std::vector<int> input_shape() override { return inp_bindshape_; }
  virtual std::vector<int> output_shape() override { return out_bindshape_; }

  virtual CanvasOutput forward(const nvtype::half* transfusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward({/* input  */ transfusion_feature,
                      /* output */ out_bindings_}, _stream);

    checkRuntime(cudaStreamSynchronize(_stream)); // wait for device to finish the copy
    checkRuntime(cudaMemcpyAsync(output_host_map_, out_bindings_, out_bindshape_[0] * out_bindshape_[1] * out_bindshape_[2] * out_bindshape_[3] * sizeof(half), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream)); // wait for device to finish the copy

    CanvasOutput output;
    int width = out_bindshape_[3];
    int height = out_bindshape_[2];
    int channels = out_bindshape_[1];
    int batch_size = out_bindshape_[0];
    output.resize(batch_size);
    for (int b = 0; b < batch_size; b++) {
        output[b].resize(channels);
        for (int c = 0; c < channels; c++) {
            output[b][c].resize(height);
            for (int h = 0; h < height; h++) {
                output[b][c][h].resize(width);
                for (int w = 0; w < width; w++) {
                    int idx = b * (channels * height * width) + c * (height * width) + h * width + w;
                    // std::cout<< "calue is: " << output_host_map_[idx] << std::endl;
                    output[b][c][h][w] = __half2float(output_host_map_[idx]);
                }
            }
        }
    }

    return output;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  half* out_bindings_;
  std::vector<int> out_bindshape_;
  std::vector<int> inp_bindshape_;
  MapSegHeadParameter param_;
  half* output_host_map_ = nullptr;

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

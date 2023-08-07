#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-mapsegm.hpp"
#include <iostream>

namespace bevfusion {
namespace head {
namespace mapsegm {

class MapSegHeadImplement : public MapSegHead {
 public:
  virtual ~MapSegHeadImplement() {
    for (size_t i = 0; i < bindings_.size(); ++i) checkRuntime(cudaFree(bindings_[i]));

    if (output_host_map_) checkRuntime(cudaFreeHost(output_host_map_));
  }

  virtual bool init(const MapSegHeadParameter& param) {
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    param_ = param;
    create_binding_memory();
    std::cout << "Binding shape is: " << bindshape_[0][0] << " " << bindshape_[0][1] << " " << bindshape_[0][2] << " "
              << bindshape_[0][3] << std::endl;
    checkRuntime(cudaMallocHost(&output_host_map_, bindshape_[0][0] * bindshape_[0][1] * bindshape_[0][2] * bindshape_[0][3] * sizeof(float)));
    return true;
  }

  void create_binding_memory() {
    std::cout<<"create binding memory"<<std::endl;
    std::cout<<"engine_->num_bindings() is: "<<engine_->num_bindings()<<std::endl;
    for (int ibinding = 0; ibinding < engine_->num_bindings(); ++ibinding) {
      if (engine_->is_input(ibinding)) continue;

      auto shape = engine_->static_dims(ibinding);
      Asserts(engine_->dtype(ibinding) == TensorRT::DType::HALF, "Invalid binding data type.");

      size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      half* pdata = nullptr;
      checkRuntime(cudaMalloc(&pdata, volumn * sizeof(half)));

      bindshape_.push_back(shape);
      bindings_.push_back(pdata);
    }
    Assertf(bindings_.size() == 1, "Invalid output num of bindings[%d]", static_cast<int>(bindings_.size()));
  }

  virtual void print() override { engine_->print("MapSegHead"); }

  virtual CanvasOutput forward(const nvtype::half* transfusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward({/* input  */ transfusion_feature,
                      /* output */ bindings_[0]}, _stream);

    checkRuntime(cudaMemcpyAsync(output_host_map_, bindings_[0], bindshape_[0][0] * bindshape_[0][1] * bindshape_[0][2] * bindshape_[0][3] * sizeof(half), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream)); // wait for device to finish the copy

    CanvasOutput output;
    int width = bindshape_[0][3];
    int height = bindshape_[0][2];
    int channels = bindshape_[0][1];
    int batch_size = bindshape_[0][0];
    output.resize(batch_size);
    printf("width is: %d, height is: %d, channels is: %d, batch_size is: %d\n", width, height, channels, batch_size);
    for (int b = 0; b < batch_size; b++) {
        output[b].resize(channels);
        for (int c = 0; c < channels; c++) {
            output[b][c].resize(height);
            for (int h = 0; h < height; h++) {
                output[b][c][h].resize(width);
                for (int w = 0; w < width; w++) {
                    int idx = b * (channels * height * width) + c * (height * width) + h * width + w;
                    output[b][c][h][w] = output_host_map_[idx];
                }
            }
        }
    }

    return output;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<half*> bindings_;
  std::vector<std::vector<int>> bindshape_;
  MapSegHeadParameter param_;
  float* output_host_map_ = nullptr;
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



#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "lidar-decoder.hpp"
#include <cfloat>
#include <limits>

namespace bevfusion {
namespace fuser {

class LidarDecoderImplement : public LidarDecoder {
 public:
  virtual ~LidarDecoderImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  virtual bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    int output_binding = 1;
    auto shape = engine_->static_dims(output_binding);
    Asserts(engine_->dtype(output_binding) == TensorRT::DType::HALF, "Invalid binding data type.");

    size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volumn * sizeof(half)));
    return true;
  }

  virtual void print() override { engine_->print("LidarDecoder"); }

  virtual nvtype::half* forward(const nvtype::half* lidar_bev, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward({/* input  */ lidar_bev,
                      /* output */ output_},
                     _stream);



    return output_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* output_ = nullptr;
  std::vector<std::vector<int>> bindshape_;
  float* lidar_feature_ = nullptr;
  std::vector<int> lidar_bindshape_;
};

std::shared_ptr<LidarDecoder> create_lidardecoder(const std::string& param) {
  std::shared_ptr<LidarDecoderImplement> instance(new LidarDecoderImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace fuser
};  // namespace bevfusion
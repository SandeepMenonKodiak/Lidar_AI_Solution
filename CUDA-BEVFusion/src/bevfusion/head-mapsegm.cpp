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
    checkRuntime(cudaMallocHost(&output_host_map_, bindshape_[0][2] * bindshape_[0][3] * sizeof(float)));
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

    checkRuntime(cudaMemcpyAsync(output_host_map_, bindings_[0], bindshape_[0][2] * bindshape_[0][3] * sizeof(half),
                                 cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    CanvasOutput output;
    // Here we assume the model output is a 2D map of floats, you'll need to adjust the conversion according to the actual model output format
    for (int i = 0; i < bindshape_[0][2]; i++) {
      std::vector<float> row(bindshape_[0][3]);
      for (int j = 0; j < bindshape_[0][3]; j++) {
        row[j] = output_host_map_[i * bindshape_[0][3] + j];
      }
      output.push_back(row);
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


#ifndef __HEAD_MAPREGR_HPP__
#define __HEAD_MAPREGR_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace mapregr {


struct MapRegrHeadParameter {
  std::string model;
  int interpolation_mode=0;
  int padding_mode = 0;
  bool align_corners = false;
  
};

using RegrOutput = std::vector<std::vector<std::vector<float>>>;

class MapRegrHead {
 public:
  virtual RegrOutput forward(const nvtype::half* transfusion_feature, void* stream) = 0;
  virtual void print() = 0;
  virtual std::vector<int> input_shape() = 0;
  virtual std::vector<int> output_shape() = 0;
};

std::shared_ptr<MapRegrHead> create_mapregrhead(const MapRegrHeadParameter& param);

};  // namespace mapregr
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_MAPREGR_HPP__

#ifndef __HEAD_MAPSEGM_HPP__
#define __HEAD_MAPSEGM_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace mapsegm {


struct MapSegHeadParameter {
  std::string model;
  int interpolation_mode=0;
  int padding_mode = 0;
  bool align_corners = false;
  
};

using CanvasOutput = std::vector<std::vector<std::vector<std::vector<float>>>>;

class MapSegHead {
 public:
  virtual CanvasOutput forward(const nvtype::half* transfusion_feature, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<MapSegHead> create_mapseghead(const MapSegHeadParameter& param);

};  // namespace mapsegm
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_MAPSEGM_HPP__
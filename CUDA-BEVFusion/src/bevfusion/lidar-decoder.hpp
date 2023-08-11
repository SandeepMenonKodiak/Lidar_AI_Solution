
#ifndef __LIDARDECODER_HPP__
#define __LIDARDECODER_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace fuser {

class LidarDecoder {
 public:
  virtual nvtype::half* forward(const nvtype::half* lidar_bev, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<LidarDecoder> create_lidardecoder(const std::string& model);

};  // namespace fuser
};  // namespace bevfusion

#endif  // __LIDARDECODER_HPP__
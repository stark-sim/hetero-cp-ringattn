#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hcp_ringattn {

enum class TensorDType {
  kUnknown = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
  kInt32 = 4,
  kInt64 = 5,
};

// 这里保留一个最小通用 tensor 容器，作为 HCP 低边界消息的载体。
// 它不再复用 HLPP 的跨域 batch 语义，只表达“一个命名的张量块”。
struct BoundaryTensor {
  std::string name;
  TensorDType dtype = TensorDType::kUnknown;
  std::string layout;
  std::vector<int64_t> shape;
  std::vector<uint8_t> payload;
};

}  // namespace hcp_ringattn


#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hcp_ringattn/core/tensor_types.h"

namespace hcp_ringattn {

// 一次 ring 传递的 K/V 小块。
// 语义上它属于 attention 内部数据流，不再带有 HLPP 那类高边界 batch 含义。
struct RingAttnBlock {
  int64_t global_offset = 0;
  int64_t block_len = 0;
  BoundaryTensor tensor;
};

// online softmax 的运行态。
// running_max / running_sum / output 都是“当前 Q chunk”对应的局部聚合状态。
struct RingAttnSoftmaxState {
  int64_t seq_len = 0;
  int64_t num_heads = 0;
  int64_t head_dim = 0;
  BoundaryTensor running_max;
  BoundaryTensor running_sum;
  BoundaryTensor output;
};

enum class RingAttnMessageType : uint32_t {
  kKvBlock = 1,
  kSoftmaxState = 2,
  kTerminate = 3,
};

struct RingAttnMessage {
  uint64_t sequence_id = 0;
  int32_t layer_index = 0;
  int32_t source_domain_id = 0;
  int32_t target_domain_id = 0;
  RingAttnMessageType type = RingAttnMessageType::kKvBlock;
  std::vector<uint8_t> payload;
};

struct RingAttnDomainConfig {
  std::string domain_id;
  std::string host;
  int port = 0;
  int64_t seq_chunk_len = 0;
  int64_t block_size = 0;
  std::string device;
};

struct RingAttnConfig {
  int64_t global_seq_len = 0;
  int64_t num_heads = 0;
  int64_t head_dim = 0;
  TensorDType dtype = TensorDType::kFloat32;
  std::vector<RingAttnDomainConfig> domains;
};

}  // namespace hcp_ringattn


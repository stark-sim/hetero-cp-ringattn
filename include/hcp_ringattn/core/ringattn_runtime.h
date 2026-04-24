#pragma once

#include <memory>
#include <string>

#include "hcp_ringattn/core/ringattn_protocol.h"
#include "hcp_ringattn/core/status.h"

namespace hcp_ringattn {

// 单个 domain 在 HCP / Ring Attention 中的域内黑盒接口。
// 当前 repo 只定义“跨域低边界合同”，不规定域内 kernel 或 runtime 如何实现。
class RingAttnRuntime {
 public:
  virtual ~RingAttnRuntime() = default;

  virtual Status Init(const RingAttnDomainConfig& config,
                      const RingAttnConfig& global_config) = 0;

  virtual Status RunRingAttentionLayer(
      int layer_index,
      const RingAttnSoftmaxState& initial_state,
      RingAttnSoftmaxState* out_state) = 0;

  virtual Status Shutdown() = 0;
};

std::unique_ptr<RingAttnRuntime> CreateRingAttnRuntime(const std::string& device_type);

}  // namespace hcp_ringattn


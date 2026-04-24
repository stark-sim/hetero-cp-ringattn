#include "hcp_ringattn/core/ringattn_runtime.h"

#include <iostream>

namespace hcp_ringattn {

namespace {

// 最小占位实现：只打印日志，不执行真实计算或通信
class NoOpRingAttnRuntime : public RingAttnRuntime {
 public:
  Status Init(const RingAttnDomainConfig& config,
              const RingAttnConfig& global_config) override {
    if (config.domain_id.empty()) {
      return Status::Error(Status::Code::kInvalidArgument,
                           "domain_id must not be empty");
    }
    if (config.seq_chunk_len <= 0) {
      return Status::Error(Status::Code::kInvalidArgument,
                           "seq_chunk_len must be positive for domain=",
                           config.domain_id);
    }
    if (config.block_size <= 0) {
      return Status::Error(Status::Code::kInvalidArgument,
                           "block_size must be positive for domain=",
                           config.domain_id);
    }
    if (global_config.global_seq_len <= 0 || global_config.num_heads <= 0 ||
        global_config.head_dim <= 0) {
      return Status::Error(Status::Code::kInvalidArgument,
                           "global config is incomplete");
    }
    config_ = config;
    global_config_ = global_config;
    std::cout << "[NoOpRingAttnRuntime] Init domain=" << config.domain_id
              << " seq_chunk=" << config.seq_chunk_len
              << " block_size=" << config.block_size
              << " device=" << config.device << "\n";
    return Status::Ok();
  }

  Status RunRingAttentionLayer(
      int layer_index,
      const RingAttnSoftmaxState& initial_state,
      RingAttnSoftmaxState* out_state) override {
    if (out_state == nullptr) {
      return Status::Error(Status::Code::kInvalidArgument,
                           "out_state must not be null");
    }
    std::cout << "[NoOpRingAttnRuntime] Run layer=" << layer_index
              << " domain=" << config_.domain_id << "\n";
    *out_state = initial_state;
    return Status::Ok();
  }

  Status Shutdown() override {
    std::cout << "[NoOpRingAttnRuntime] Shutdown domain=" << config_.domain_id << "\n";
    return Status::Ok();
  }

 private:
  RingAttnDomainConfig config_;
  RingAttnConfig global_config_;
};

}  // namespace

std::unique_ptr<RingAttnRuntime> CreateRingAttnRuntime(const std::string& device_type) {
  (void)device_type;
  return std::make_unique<NoOpRingAttnRuntime>();
}

}  // namespace hcp_ringattn

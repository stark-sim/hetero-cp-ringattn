#include <iostream>
#include <vector>

#include "hcp_ringattn/core/ringattn_runtime.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::cout << "=== RingAttn Coordinator Smoke ===\n";

  hcp_ringattn::RingAttnConfig global_config;
  global_config.global_seq_len = 1024;
  global_config.num_heads = 8;
  global_config.head_dim = 64;
  global_config.dtype = hcp_ringattn::TensorDType::kFloat32;

  std::vector<hcp_ringattn::RingAttnDomainConfig> domains = {
      {"domain-0", "127.0.0.1", 26001, 512, 128, "cpu"},
      {"domain-1", "127.0.0.1", 26002, 256, 64, "cpu"},
      {"domain-2", "127.0.0.1", 26003, 256, 64, "cpu"},
  };
  global_config.domains = std::move(domains);

  int pass_count = 0;
  for (const auto& domain_cfg : global_config.domains) {
    auto runtime = hcp_ringattn::CreateRingAttnRuntime("cpu");
    auto status = runtime->Init(domain_cfg, global_config);
    if (!status.ok()) {
      std::cerr << "Init failed for " << domain_cfg.domain_id
                << ": " << status.message() << "\n";
      return 1;
    }

    hcp_ringattn::RingAttnSoftmaxState initial_state;
    hcp_ringattn::RingAttnSoftmaxState out_state;
    status = runtime->RunRingAttentionLayer(0, initial_state, &out_state);
    if (!status.ok()) {
      std::cerr << "Run failed for " << domain_cfg.domain_id
                << ": " << status.message() << "\n";
      return 1;
    }

    status = runtime->Shutdown();
    if (!status.ok()) {
      std::cerr << "Shutdown failed for " << domain_cfg.domain_id
                << ": " << status.message() << "\n";
      return 1;
    }
    ++pass_count;
  }

  std::cout << "=== RingAttn Coordinator Smoke PASSED ===\n";
  std::cout << "domains=" << pass_count << "/" << global_config.domains.size() << "\n";
  return 0;
}
